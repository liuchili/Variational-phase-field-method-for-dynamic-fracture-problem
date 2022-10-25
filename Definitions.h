// -*- C++ -*-
#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
using std::sqrt;
using std::sin;
using std::cos;
using std::isnan;
using std::pow;
using std::cbrt;
using std::fabs;
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>		// file io
using std::stringstream;
using std::ifstream;
using std::getline;
using std::istringstream;
#include <iomanip>
#include <cassert>
#include <iostream>
#include <set>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <float.h>		// DBL_MAX, INT_MAX
#include <math.h>
#include <mpi.h>
#include<omp.h>

#include <chrono>
using namespace std::chrono;

#include<complex>
using std::complex;

#include <numeric>
using std::accumulate;
using std::cout;
using std::endl;

#include <algorithm>		// math stuff
using std::min;
using std::max;
using std::find;
using std::distance;

#include <limits>
#include <limits>
#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif
#include <cstddef>
using std::numeric_limits;


#include <stdlib.h>     // atoi

#include <vector>  // standard vector
#include <array>
using std::vector;
using std::array;

//#include<tr1/array>
//using std::tr1::array;
using std::string;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;

#include<Eigen/SparseLU>
#include<Eigen/SparseQR>
#include <Eigen/OrderingMethods>
using Eigen::SparseLU;
using Eigen::SparseQR;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::EigenSolver;
using Eigen::ConjugateGradient;
using Eigen::BiCGSTAB;
using Eigen::RowMajor;
using Eigen::Lower;
using Eigen::Upper;
#endif // DEFINITIONS_H
