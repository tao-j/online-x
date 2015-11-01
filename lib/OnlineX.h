#ifndef ONLINEX_ONLINEX_H
#define ONLINEX_ONLINEX_H

#include "util.h"
#include "objcokus.h"

class OnlineX {
public:
    OnlineX(const char**);
    ~OnlineX();

    void create(vector<vector <int>> item_text);
    double feed_one(int user_i, int item_j, double rating); // returns accumulated train time
    double eval_by_theta(int, int);
    double eval_by_theta_inferred(int, int);
    // hyper parameters
    unsigned int K;     // number of topics.
    unsigned int T;     // number of total words.
    int U;            // number of total users
    int V;            // number of total items
    int I;            // number of mean-field rounds for each Bayesian update.
    int J;            // number of Gibbs samples in the mean-field update of latent variables (substract Jb)
    int Jb;           // number of burn-in steps for Gibbs samples of latent variables.
    double a0;    // prior of document topic distribution.
    double b0;     // prior of word topic distribution.
    double c;         // learning rate.
    double e;         // not used
    double su;         // prior weight \sim N(0, v^2).
    double sv;         // prior weight \sim N(0, v^2).
    double lu;
    double lv;

    vector<vector<double> > phi;     // sufficient statistics.
    vector<double> phi_sum;          // sums of rows in phi.

    vector<vector<double> > u_cov;
    vector<vector<double> > v_cov;
    vector<vector<double> > u_mean;
    vector<vector<double> > v_mean;

    int user_i, item_j;
    double r, r_hat, delta;

    vector<vector<int> > doc;
    vector<vector<int> > Z;
    vector<vector<double> > Cdk;

    /* results */
    double train_time;

    /* source of randomness */
    objcokus cokus;

    /* temp variable */
    vector<double> new_v_mean, new_u_mean;
    vector<double> new_v_cov, new_u_cov;
    vector<vector<double> > stat_phi;              // stat used in global update.
    vector<int> stat_phi_list_k, stat_phi_list_t;  // aux stats for sparse updates.

    /* logical */
    string algo;
    bool is_ostmr, is_osgd_topic, is_ocf_cov, is_pa_i;
    bool run_tm, is_tm_w_v, run_cf;
    bool run_load_cdk, run_save_cdk, run_save_phi, run_save_v_mean, run_save_u_mean;

    void collect_phi(bool);
    void calc_phi(bool, int);
    void uv_pa_i();
    void uv_osgd_topic();
    void uv_ocf_cov();
    // Online Simultaneous Topic Modeling and Regression
    void uv_ostmr();
    void update_tm();
    void update_z_joint();
    void update_z_sole();
    double eval_log_likelihood(int);
    double eval_one(int, int);
};

#endif //ONLINEX_ONLINEX_H
