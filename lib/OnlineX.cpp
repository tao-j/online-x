#include <assert.h>
#include <stdlib.h>
#include "OnlineX.h"

using namespace std;

OnlineX::OnlineX(const char** argv) {
    is_ostmr = is_osgd_topic = is_ocf_cov = is_pa_i = false;
    run_tm = is_tm_w_v = false;
    run_load_cdk = run_save_cdk = run_save_phi = run_save_v_mean = run_save_u_mean = false;

/* 2
 * algo[ofm] K U V T I J Jb e c a0 b0 lu(lambda_u) lv(lambda_v) su sigma_v
 * algo? update_z? update_uv? save_uv? save_gamma? save_cdk? load_cdk?
 *
 * algo-what_stat-#k-hash.log[.txt]
 */
    algo = string(argv[1]);
    K = atoi(argv[2]);
    U = atoi(argv[3]);
    V = atoi(argv[4]);
    T = atoi(argv[5]);
    I = atoi(argv[6]);
    J = atoi(argv[7]);
    Jb= atoi(argv[8]);
    e = atof(argv[9]);
    c = atof(argv[10]);
    a0= atof(argv[11]);
    b0= atof(argv[12]);
    lu= atof(argv[13]);
    lv= atof(argv[14]);
    su= atof(argv[15]);
    sv= atof(argv[16]);

    // algo
    if (algo == "gibbs-lda") {
        run_tm = true;
        run_save_cdk = true;
        run_save_phi = true;
    }
    else if (algo == "pa-i") {
        run_cf = true;
        is_pa_i = true;
    }
    else if (algo == "ocf-cov") {
        run_cf = true;
        is_ocf_cov = true;
    }
    else if (algo == "osgd-topic-fixed") {
        run_cf = true;
        is_osgd_topic = true;
        run_load_cdk = true;
        run_save_u_mean = true;
        run_save_v_mean = true;
    }
    else if (algo == "osgd-topic-flow") {
        run_tm = true;
        run_cf = true;
        is_osgd_topic = true;
        run_save_cdk = true;
        run_save_phi = true;
        run_save_u_mean = true;
        run_save_v_mean = true;
    }
    else if (algo == "ostmr") {
        run_tm = true;
        run_cf = true;
        is_ostmr = true;
        is_tm_w_v = true;
        run_save_cdk = true;
        run_save_phi = true;
        run_save_u_mean = true;
        run_save_v_mean = true;
    }
}

OnlineX::~OnlineX() {
}

void OnlineX::create(vector<vector<int> > item_text) {
    // init random source
    cokus.reloadMT();
    cokus.seedMT((uint32)time(NULL));

    train_time = 0;

    doc = vector<vector<int> >(item_text);
    assert(doc.size() == V);

    Z.resize(doc.size());
    for (int vi = 0; vi < doc.size(); ++vi) {
        Z[vi].resize(doc[vi].size(), 0);
    }
    Cdk.resize(V, vector<double>(K, 0));
    for(int d = 0; d < V; d++) {
        for(int w = 0; w < doc[d].size(); w++) {
            // assign each word a random topic
            //TODO: fix signed or unsigned?
            Z[d][w] = (int)(cokus.randomMT() % K + K) % K;
            // update count statistics
            ++Cdk[d][Z[d][w]];
        }
    }

    // init U V (Gaussian posterior) according to Gaussian prior
    u_cov.resize(U, vector<double>(K, 0.));
    u_mean.resize(U, vector<double>(K, 0.));

    for (int ui = 0; ui < U; ui++) {
        for(int k = 0; k < K; k++) {
            u_cov[ui][k] = su * su;
            u_mean[ui][k] = cokus.random01() * 1.;
        }
    }
    v_cov.resize(V, vector<double>(K, 0.));
    v_mean.resize(V, vector<double>(K, 0.));
    for (int vi = 0; vi < V; vi++) {
        for(int k = 0; k < K; k++) {
            v_cov[vi][k] = sv * sv;
            v_mean[vi][k] = cokus.random01() * 1.;
        }
    }

    // init \phi
    phi = vector<vector<double> >(K);
    stat_phi = vector<vector<double> >(K);
    for(int k = 0; k < K; k++) {
        phi[k].resize(T, 0);
        stat_phi[k].resize(T, 0);
    }
    phi_sum = vector<double>(K, 0);
    stat_phi_list_k = vector<int>();
    stat_phi_list_t = vector<int>();
}

void OnlineX::update_z_joint() {

    double weights[K];
    int word_id;
    size_t N;
    double sel, cul;
    int seli;
    N = doc[item_j].size();
    for (int wi = 0; wi < N; wi++) {
        word_id = doc[item_j][wi];

        Cdk[item_j][Z[item_j][wi]]--; // exclude topic assignment for Zjn.

        int flagZ = -1, flag0 = -1; // flag for abnormality.
        for(int k = 0; k < K; k++) {
            if(k == 0) cul = 0;
            else cul = weights[k-1];
            weights[k] = cul+(Cdk[item_j][k]+ a0)
                 /* strategy 1 variational optimal distribution */
                 // *exp(digamma(b0+gamma[k][word])-digamma(b0*T+gammasum[k]))
                 /* strategy 2 approximation that does not require digamma() */
                 *(b0 +phi[k][word_id])/(b0 *T+phi_sum[k])
                 *exp(0.5/ sv / sv /(double)N*(2.0*v_mean[item_j][k]-(1.0+2.0*Cdk[item_j][k])/(double)N))
                ;
            assert(!std::isnan(weights[k]) && "error: Z weights nan.\n");
            if(std::isinf(weights[k])) flagZ = k; // too discriminative, directly set val.
            if(weights[k] > 0) flag0 = 1;  // it must be >0 if no err, e^x should always >0
        } // for each topic
        if(flagZ >= 0) Z[item_j][wi] = flagZ;
        else if(flag0 == -1) Z[item_j][wi] = (int)(cokus.randomMT() % K + K) % K;
        else {
            sel = weights[K-1] * cokus.random01();
            for(seli = 0; weights[seli] < sel; seli++);
            Z[item_j][wi] = seli;
        }
        Cdk[item_j][Z[item_j][wi]]++; // restore Cdk
    } // for each word
}

void OnlineX::update_z_sole() {

    double weights[K];
    int word_id;
    int N;
    double sel, cul;
    int seli;
    N = doc[item_j].size();
    for (int wi = 0; wi < N; wi++) {
        word_id = doc[item_j][wi];

        Cdk[item_j][Z[item_j][wi]]--; // exclude topic assignment for Zjn.

        int flagZ = -1, flag0 = -1; // flag for abnormality.
        for(int k = 0; k < K; k++) {
            if(k == 0) cul = 0;
            else cul = weights[k-1];
            weights[k] = cul+(Cdk[item_j][k]+ a0)
                             /* strategy 1 variational optimal distribution */
                             // *exp(digamma(b0+gamma[k][word])-digamma(b0*T+gammasum[k]))
                             /* strategy 2 approximation that does not require digamma() */
                             *(b0 +phi[k][word_id])/(b0 *T+phi_sum[k])
                    ;
            assert(!std::isnan(weights[k]) && "error: Z weights nan.\n");
            if(std::isinf(weights[k])) flagZ = k; // too discriminative, directly set val.
            if(weights[k] > 0) flag0 = 1;  // it must be >0 if no err, e^x should always >0
        } // for each topic
        if(flagZ >= 0) Z[item_j][wi] = flagZ;
        else if(flag0 == -1) Z[item_j][wi] = (int)(cokus.randomMT() % K + K) % K;
        else {
            sel = weights[K-1] * cokus.random01();
            for(seli = 0; weights[seli] < sel; seli++);
            Z[item_j][wi] = seli;
        }
        Cdk[item_j][Z[item_j][wi]]++; // restore Cdk
    } // for each word
}

void OnlineX::calc_phi(bool remove, int rounds) {

    for(size_t stat_i = 0; stat_i < stat_phi_list_k.size(); stat_i++) {
        int k = stat_phi_list_k[stat_i], t = stat_phi_list_t[stat_i];

        phi[k][t] += stat_phi[k][t] / (double) rounds;
        phi_sum[k] += stat_phi[k][t] / (double) rounds;
    }

}

void OnlineX::collect_phi(bool reset) {
    if(reset) {
        stat_phi.clear();
        stat_phi.resize(K, vector<double>(T, 0.));
        stat_phi_list_k.clear();
        stat_phi_list_t.clear();
    }

    // collect temporary patch to \phi
    // according to Z_ij record how many times one word is reached
    for(int i = 0; i < doc[item_j].size(); i++) { // for each word in a document
        int k = Z[item_j][i], word_id = doc[item_j][i];
        if(stat_phi[k][word_id] == 0) {
            stat_phi_list_k.push_back(k);
            stat_phi_list_t.push_back(word_id);
        }
        ++stat_phi[k][word_id];
    }
}

void OnlineX::update_tm() {
    for (int j = 0; j < J; ++j) {
        if (is_tm_w_v) {
            update_z_joint();
        }
        else {
            update_z_sole();
        }

        if (j < Jb) continue;
        collect_phi(j == Jb);
    }
    calc_phi(false, J - Jb);
}

void OnlineX::uv_pa_i() {
    double power_sum_u = 0;
    double power_sum_v = 0;
    double dir = 0.;
    if (delta > 0.) dir = 1.; else dir = -1.;
    double loss = abs(delta);
    for (int i = 0; i < K; i++ ) power_sum_u += u_mean[user_i][i];
    for (int i = 0; i < K; i++ ) power_sum_v += v_mean[item_j][i];
    double tao_u = min(c, loss / power_sum_u);
    double tao_v = min(c, loss / power_sum_v);

    for (int k = 0; k < K; ++k) {
        double temp = 0.;
        temp += dir * tao_v * v_mean[item_j][k];
        v_mean[item_j][k] += dir * tao_u * u_mean[user_i][k];
        u_mean[user_i][k] += temp;
    }
}

void OnlineX::uv_osgd_topic() {

    for (int k = 0; k < K; ++k) {
        new_u_mean[k] = u_mean[user_i][k] - c*(lu * u_mean[user_i][k] - delta * v_mean[item_j][k]);
        // TODO: make it theta
        new_v_mean[k] = v_mean[item_j][k] - c*(lv * (v_mean[item_j][k] - 0*Cdk[item_j][k]/doc[item_j].size()) - delta * u_mean[user_i][k]);
    }
    for (int k = 0; k < K; k++) {
        u_mean[user_i][k] = new_u_mean[k];
        v_mean[item_j][k] = new_v_mean[k];
    }

}

void OnlineX::uv_ocf_cov() {

    double denomi_u, denomi_v;
    denomi_u = lu * lu;
    denomi_v = lu * lu;
    for (int k = 0; k < K; k++) {
        denomi_u += (v_mean[item_j][k] * u_cov[user_i][k] * v_mean[item_j][k]);
        denomi_v += (u_mean[user_i][k] * v_cov[item_j][k] * u_mean[user_i][k]);
    }

    for (int k = 0; k < K; ++k) {
        u_cov[user_i][k] = 1./(1./u_cov[user_i][k] + v_mean[item_j][k]*v_mean[item_j][k]/ lv / lv);
        u_mean[user_i][k] += delta / denomi_u * u_cov[user_i][k] * v_mean[item_j][k];
        v_cov[item_j][k] = 1./(1./v_cov[item_j][k]+u_mean[user_i][k]*u_mean[user_i][k]/ lv / lv);
        v_mean[item_j][k] += delta / denomi_v * v_cov[item_j][k] * u_mean[user_i][k];
    }

}

void OnlineX::uv_ostmr() {

    double denomi_u, denomi_v;
    denomi_u = lu * lu;
    denomi_v = lu * lu;
    for (int k = 0; k < K; k++) {
        denomi_u += (v_mean[item_j][k] * u_cov[user_i][k] * v_mean[item_j][k]);
        denomi_v += (u_mean[user_i][k] * v_cov[item_j][k] * u_mean[user_i][k]);
    }

    for (int k = 0; k < K; ++k) {
        new_u_cov[k] = 1./(1./u_cov[user_i][k]+v_mean[item_j][k]*v_mean[item_j][k]/ lv / lv);
        new_u_mean[k] = delta / denomi_u * u_cov[user_i][k] * v_mean[item_j][k];
        new_v_cov[k] = 1./(1./v_cov[item_j][k]+1./ sv / sv +u_mean[user_i][k]*u_mean[user_i][k]/ lv / lv);
    }

    vector<double> v_cov_mix;
    v_cov_mix.resize(K, 0);
    for (int k = 0; k < K; k++ ) {
        v_cov_mix[k] = 1./(1./v_cov[item_j][k] + 1./ sv / sv);
    }

    double denomi_mix, nomi_op1, nomi_op2;
    denomi_mix = 1.; nomi_op1 = nomi_op2 = 0.;
    for (int k = 0; k < K; k++) {
        nomi_op1 += u_mean[user_i][k]* v_cov_mix[k]/v_cov[item_j][k]*v_mean[item_j][k];
        nomi_op2 += u_mean[user_i][k]* v_cov_mix[k]/ sv / sv *Cdk[item_j][k]/doc[item_j].size();
        denomi_mix += u_mean[user_i][k]* v_cov_mix[k]/ lv / lv *u_mean[user_i][k];
    }
    double mult = (nomi_op1 + 1.*nomi_op2 - r) / denomi_mix;

    for (int k = 0; k < K; k++) {
        v_mean[item_j][k] = v_cov_mix[k]/v_cov[item_j][k]*v_mean[item_j][k] +
                            1.* v_cov_mix[k]/ sv / sv *Cdk[item_j][k]/doc[item_j].size() -
                            v_cov_mix[k]/ lu / lu *u_mean[user_i][k]*mult;
        u_cov[user_i][k] = new_u_cov[k];
        u_mean[user_i][k] += new_u_mean[k];
        v_cov[item_j][k] = new_v_cov[k];
    }
}

double OnlineX::eval_log_likelihood(int item_j) {

    // evaluate log likelihood for given document

    vector<double> theta, beta;
    theta.resize(K, 0); beta.resize(K, 0);
    double Cdk_sum = 0.;
    for (int k = 0; k < K; k ++) {
        Cdk_sum += Cdk[item_j][k];
    }
    for (int k = 0; k < K; k ++) {
        theta[k]= (Cdk[item_j][k] + a0) / (a0 * K + Cdk_sum);
    }

    double temp;
    double score = 0;
    for (int wi = 0; wi < doc[item_j].size(); wi++) {

        double probability = 0; // chance of this word's appearance
        for (int k = 0; k < K; k++) {
            int t = doc[item_j][wi];
            temp = (b0 + phi[k][t]) / (b0 * T + phi_sum[k]);
            probability += temp * theta[k];
        }
        score += log(probability);
    }

    return score / doc[item_j].size();
}

double OnlineX::eval_one(int uid, int vid) {
    return dotp(u_mean[uid], v_mean[vid]);
}

double OnlineX::eval_by_theta(int uid, int vid) {
    double ret = 0.;
    for (int k = 0; k < K; ++k) {
        ret += u_mean[uid][k] * Cdk[vid][k] / doc[vid].size();
    }
    return ret;
}

double OnlineX::eval_by_theta_inferred(int uid, int vid) {
    item_j = vid;
    for (int i = 0; i < Jb; ++i) {
        update_z_sole();
    }
    return eval_by_theta(uid, vid);
}

double OnlineX::feed_one(int ui, int ij, double rat) {
    user_i = ui; item_j = ij; r = rat;
    r_hat = dotp(u_mean[user_i], v_mean[item_j]);
    delta = r - r_hat;
    new_v_mean.resize(K, 0.); new_u_mean.resize(K, 0.);
    new_v_cov.resize(K, 0.); new_u_cov.resize(K, 0.);

    clock_t time_start = clock();
    clock_t time_end;


    if (run_tm) {
        update_tm();
    }

    if (run_cf) {
        if (is_ostmr) {
            uv_ostmr();
        } else if (is_osgd_topic) {
            uv_osgd_topic();
        } else if (is_ocf_cov) {
            uv_ocf_cov();
        } else if (is_pa_i) {
            uv_pa_i();
        } else {
            cerr << "error: no algo chosen" << endl;
        }
    }

    time_end = clock();
    train_time += (double)(time_end-time_start)/CLOCKS_PER_SEC;
    return train_time;
}
