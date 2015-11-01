#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "OnlineX.h"
#include <ctime>
#include <stdlib.h>

using namespace std;

int main (int argc, const char** argv) {

    if (argc != 25) {
        cerr << "provide data direcotry correctly by passing arguments. exiting." << endl;
        return -1;
    }

    // file name pre/suffix
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,80,"%m%d-%H%M%S", timeinfo);
    string suffix(buffer);
    string prefix = string(argv[1]);
    string algo = string(argv[1]);

    // init class
    OnlineX xctr(argv);

    // train test files
    string  train(argv[17]);
    string   test(argv[18]);
    int learn_cnt= atoi(argv[19]);
    int test_cnt = atoi(argv[20]);
    string   basedir(argv[21]);
    int   test_interval= atoi(argv[22]);
    string cdk_file(argv[23]);
    string ofm_method(argv[24]);

    bool in_matrix, run_ofm_infer;
    in_matrix = run_ofm_infer = false;
    if (ofm_method == "ofm-static") {
        run_ofm_infer = false;
        suffix += "ofs";
    } else if (ofm_method == "ofm-infer") {
        run_ofm_infer = true;
        suffix += "ofi";
    } else if (ofm_method == "in-matrix") {
        in_matrix = true;
    } else {
        cerr << "ofm method undefined" << endl;
        return -1;
    }

    ifstream fin;
    ofstream log_out;

    // save params, logging
    log_out.open(prefix + suffix + string(".log"));
    for (int i = 0; i < argc; ++i) {
        log_out << argv[i] << endl;
    }

    // read in docs
    string line;
    fin.open(basedir + string("_docs"));
    if (!fin) {
        cerr << "unable to open " << basedir + string("_docs") << " file. exiting\n";
    }
    vector<vector<int> > corpus; corpus.clear();
    while (getline(fin, line)) {
        stringstream sstream;
        sstream << line;
        vector<int> one_plot; one_plot.clear();
        int wordid;
        while (sstream >> wordid) {
            one_plot.push_back(wordid);
        }
        corpus.push_back(one_plot);
    }
    // init class
    xctr.create(corpus);
    log_out << "class created\n";
    fin.close();

    vector<int> users, items;
    vector<double> ratings;
    vector<int> users_test, items_test;
    vector<double> ratings_test;
    int user, item; double rating;
    // train data
    fin.open(basedir + train);
    if (!fin) {
        cerr << "ERROR: unable to open training " << basedir+train << " file\n";
        log_out << "ERROR: unable to open training _rats file\n";
        return -1;
    }
    for (int i = 0; i < learn_cnt; i++) {
        fin >> user >> item >> rating;
        users.push_back(user); items.push_back(item); ratings.push_back(rating);
    }
    fin.close();
    log_out << train << ": read train done: " << users.size() << endl;

    // test data
    fin.open(basedir + test);
    if (!fin) {
        cerr << "ERROR: unable to open testing _rats file\n";
        log_out << "ERROR: unable to open testing _rats file\n";
        return -1;
    }
    for (int i = 0; i < test_cnt; i++) {
        fin >> user >> item >> rating;
        users_test.push_back(user); items_test.push_back(item); ratings_test.push_back(rating);
    }
    fin.close();
    log_out << test << ": read test done: " << users_test.size() << endl;
    log_out.close();

    // algo
    bool run_test_rmse, run_train_rmse, log_likelihood;
    bool run_load_cdk, run_save_cdk, run_save_phi, run_save_v_mean, run_save_u_mean;
    run_test_rmse = run_train_rmse = log_likelihood = false;
    run_load_cdk = run_save_cdk = run_save_phi = run_save_v_mean = run_save_u_mean = false;
    if (algo == "gibbs-lda") {
        run_save_cdk = true;
        run_save_phi = true;

        log_likelihood = true;
    }
    else if (algo == "pa-i") {
        run_test_rmse = true;
        run_train_rmse = true;
    }
    else if (algo == "ocf-cov") {
        run_test_rmse = true;
        run_train_rmse = true;
    }
    else if (algo == "osgd-topic-fixed") {
        run_test_rmse = true;
        run_train_rmse = true;
        run_load_cdk = true;
        run_save_u_mean = true;
        run_save_v_mean = true;
    }
    else if (algo == "osgd-topic-flow") {
        run_test_rmse = true;
        run_train_rmse = true;
        run_save_cdk = true;
        run_save_phi = true;
        run_save_u_mean = true;
        run_save_v_mean = true;
        log_likelihood = true;
    }
    else if (algo == "ostmr") {
        run_test_rmse = true;
        run_train_rmse = true;
        log_likelihood = true;
        run_save_cdk = true;
        run_save_phi = true;
        run_save_u_mean = true;
        run_save_v_mean = true;
    } else {
        cerr << "unknown algorithm" << endl;
        return -1;
    }

    //
    if (run_load_cdk) {
        load_mat(xctr.V, xctr.K, cdk_file, xctr.Cdk);
    }
//    if (ofm_method == "ofm-static") {
//        vector<vector<double> > Cdk;
//        Cdk.resize(xctr.V, vector<double>(xctr.K, 0.));
//        load_mat(xctr.V, xctr.K, cdk_file, Cdk);
//        for (int vi = 0; vi < items_test.size(); ++vi) {
//            for (int k = 0; k < xctr.K; ++k) {
//                xctr.Cdk[items_test[vi]][k] = Cdk[items_test[vi]][k];
//            }
//        }
//    }

    // start learning
    ofstream fout;
    double train_time;
    double rmse_sum = 0.;
    for (int i = 0; i < learn_cnt; ++i) {

        if (run_train_rmse) {
            double delta = abs(ratings[i] - xctr.eval_one(users[i], items[i]));
//            mae += delta;
            rmse_sum += delta * delta;
        }

        train_time = xctr.feed_one(users[i], items[i], ratings[i]);

        if (i % test_interval == 0) {

            cout << train_time << ' ';

            fout.open(prefix + string("_time_") + suffix + string(".txt"), ios::app);
            fout << train_time << endl;
            fout.close();

            if (run_train_rmse) {
                fout.open(prefix + string("_train_") + suffix + string(".txt"), ios::app);
                fout << sqrt(rmse_sum / (i+1)) << endl;
                fout.close();

                cout << sqrt(rmse_sum / (i+1)) << ' ';
            }

            if (run_test_rmse) {

                double test_rmse = 0.;
                fout.open(prefix + string("_test_") + suffix + string(".txt"), ios::app);
                if (in_matrix) {
                    for (int ti = 0; ti < test_cnt; ti++ ) {
                        double diff = xctr.eval_one(users_test[ti], items_test[ti]) - ratings_test[ti];
                        test_rmse += diff * diff;
                    }
                } else if (run_ofm_infer) {
                    for (int ti = 0; ti < test_cnt; ti++ ) {
                        double diff = xctr.eval_by_theta_inferred(users_test[ti], items_test[ti]) - ratings_test[ti];
                        test_rmse += diff * diff;
                    }
                } else {
                    for (int ti = 0; ti < test_cnt; ti++ ) {
                        double diff = xctr.eval_by_theta(users_test[ti], items_test[ti]) - ratings_test[ti];
                        test_rmse += diff * diff;
                    }
                }

                fout << sqrt(test_rmse / test_cnt) << endl;
                fout.close();

                cout << sqrt(test_rmse / test_cnt) << ' ';
            }

            if (log_likelihood) {

                fout.open(prefix + string("_logp_") + suffix + string(".txt"), ios::app);
                double likeli_sum = 0.;
                for (int ii = 0; ii < 2000; ii++) {
                    likeli_sum += xctr.eval_log_likelihood(ii);
                }
                fout << likeli_sum / 2000 << endl;
                fout.close();

                cout << likeli_sum / 2000 << ' ';
            }

            cout << endl;
        }
    }

    if (run_save_cdk) {
        if (algo == "ostmr" && ofm_method != "in-matrix") {
            ifstream fin;
            fin.open(basedir+"_ofmitems");
            while (fin) {
                int id;
                fin >> id;
                xctr.item_j = id;
                xctr.update_z_sole(); xctr.update_z_sole();
            }
            cout << "of-ifer done\n";
        }
        save_mat(xctr.V, xctr.K, prefix + string("_cdk_") + suffix + string(".txt"), xctr.Cdk);
    }
    if (run_save_phi) {
        save_mat(xctr.K, xctr.T, prefix + string("_phi_") + suffix + string(".txt"), xctr.phi);
    }
    if (run_save_u_mean) {
        save_mat(xctr.U, xctr.K, prefix + string("_u_mean_") + suffix + string(".txt"), xctr.u_mean);
    }
    if (run_save_v_mean) {
        save_mat(xctr.V, xctr.K, prefix + string("_v_mean_") + suffix + string(".txt"), xctr.v_mean);
    }


    return 0;
}