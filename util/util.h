#ifndef ONLINEX_UTILS_H
#define ONLINEX_UTILS_H

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <cmath>

template <typename type>
void load_mat(int row, int col, string name, type &mat, bool show=true) {
    std::ifstream fin;
    fin.open(name);
    if(!fin) {
        cerr << "Failed to open file to load " << name << endl;
        return;
    }

    for (int v = 0; v < row; v++) {
        for (int k = 0; k < col; k++) {
            fin >> mat[v][k];
        }
    }
    fin.close();

    if (show)
        cout << "load mat" << name << " done\n";
}

template <typename type>
void save_mat(int row, int col, string name, type &mat, bool show=true) {
    std::ofstream fout;
    fout.open(name);
    if(!fout) {
        cerr << "Failed to open file to save mat " << name << endl;
        return;
    }

    for (int v = 0; v < row; v++) {
        for (int k = 0; k < col; k++) {
            fout << mat[v][k] << ' ';
        }
        fout << endl;
    }
    fout.close();

    if (show)
        cout << "save mat " << name << " done\n";
}

#include <cassert>
inline double dotp(const vector<double> &a, const vector<double> &b) {
    double ret =0.;
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}
#endif //ONLINEX_UTILS_H
