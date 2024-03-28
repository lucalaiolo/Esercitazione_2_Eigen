#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>
using namespace std;
using namespace Eigen;

Vector2d solveSystemLU(const Matrix2d& A, const Vector2d& b) {
    return A.fullPivLu().solve(b);
}
Vector2d solveSystemQR(const Matrix2d& A, const Vector2d& b) {
    return A.fullPivHouseholderQr().solve(b);
}

bool is_singular(const Matrix2d& A, double& condA) {
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    if (singularValuesA.minCoeff() < 1e-16) {
        return true;
    }
    condA = singularValuesA.maxCoeff()/singularValuesA.minCoeff();
    return false;
}

int main()
{
    // The exact solution of the following systems is
    Vector2d exactSol {-1.0, -1.0};

    // System 1
    Matrix2d A1 {
        {5.547001962252291e-01, -3.770900990025203e-02},
        {8.320502943378437e-01, -9.992887623566787e-01}
    };

    double condA1 = 0;

    Vector2d b1 {-5.169911863249772e-01, 1.672384680188350e-01};

    Vector2d sol1_LU;
    Vector2d sol1_QR;
    cout << "Linear system 1" << endl;
    if (!is_singular(A1,condA1)) {
        sol1_LU = solveSystemLU(A1,b1);
        sol1_QR = solveSystemQR(A1,b1);
        double relErr1_LU = (sol1_LU-exactSol).norm()/exactSol.norm();
        double relErr1_QR = (sol1_QR-exactSol).norm()/exactSol.norm();
        cout << scientific << setprecision(16) << "cond(A1): " << condA1 << endl;
        cout << "Relative error using LU decomposition: " << relErr1_LU << endl;
        cout << "Relative error using QR decomposition: " << relErr1_QR << endl << endl;
    } else {
        cout << "Matrix is singular." << endl;
    }

    // We solve the next two linear systems likewise
    // System 2
    Matrix2d A2 {
        {5.547001962252291e-01, -5.540607316466765e-01},
        {8.320502943378437e-01, -8.324762492991313e-01}
    };

    double condA2 = 0;

    Vector2d b2 {-6.394645785530173e-04, 4.259549612877223e-04};
    Vector2d sol2_LU;
    Vector2d sol2_QR;
    cout << "Linear system 2" << endl;
    if (!is_singular(A2,condA2)) {
        sol2_LU = solveSystemLU(A2,b2);
        sol2_QR = solveSystemQR(A2,b2);
        double relErr2_LU = (sol2_LU-exactSol).norm()/exactSol.norm();
        double relErr2_QR = (sol2_QR-exactSol).norm()/exactSol.norm();
        cout << "cond(A2): " << condA2 << endl;
        cout << "Relative error using LU decomposition: " << relErr2_LU << endl;
        cout << "Relative error using QR decomposition: " << relErr2_QR << endl << endl;
    } else {
        cout << "Matrix is singular." << endl;
    }

    //System 3
    Matrix2d A3 {
        {5.547001962252291e-01, -5.547001955851905e-01},
        {8.320502943378437e-01, -8.320502947645361e-01}
    };

    double condA3 = 0;

    Vector2d b3 {-6.400391328043042e-10, 4.266924591433963e-10};
    Vector2d sol3_LU;
    Vector2d sol3_QR;
    cout << "Linear system 3" << endl;
    if (!is_singular(A3,condA3)) {
        sol3_LU = solveSystemLU(A3,b3);
        sol3_QR = solveSystemQR(A3,b3);
        double relErr3_LU = (sol3_LU-exactSol).norm()/exactSol.norm();
        double relErr3_QR = (sol3_QR-exactSol).norm()/exactSol.norm();
        cout << "cond(A3): " << condA3 << endl;
        cout << "Relative error using LU decomposition: " << relErr3_LU << endl;
        cout << "Relative error using QR decomposition: " << relErr3_QR << endl << endl;
    } else {
        cout << "Matrix is singular." << endl;
    }

    return 0;
}
