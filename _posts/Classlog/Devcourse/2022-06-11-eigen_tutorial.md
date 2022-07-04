---
title:    "[데브코스] 17주차 - Visual-SLAM Eigen tutorial "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-06-11 02:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, Visual-SLAM]
toc: true
comments: true
math: true
---

<br>

# Eigen tutorial

참고 자료 : [Eigen 사이트 공식 문서](https://eigen.tuxfamily.org/dox/GettingStarted.html)

<br>

## 1. Simply Class

[Eigen 사이트 공식 문서 reference guide](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)

- Matrix2f, Matrix2d, Matrix3f, Matrix3d, MatrixXd ...
- Vector2f, Vector2d, Vector3f, Vector3d, VectorXd ...

<br>

Vector와 Matrix의 사용 형태는 거의 동일하니 Matrix에 대해서만 다루려고 한다. 기본적인 변수 선언은 다음과 같다. Zero를 통해 0으로 구성된 Matrix를 구성할 수 있고, Random을 통해 무작위의 값으로 구성할 수 있다.

```cpp
#include <iostream>
#include <Eigen/Dense>

int main()
{
  Eigen::Matrix3d m1 = Eigen::Matrix3d::Zero();
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Zero(3,3);
  Eigen::Matrix2f m3 = Eigen::Matrix2f::Random();
  Eigen::MatrixXf m4 = Eigen::Matrix2f::Random(2,2);
}
```

<br>

이런 Matrix에서의 element를 참조하는 방법은 다음과 같다.

```cpp
int main()
{
  m1(0, 1) = 3;
  m1(2, 2) = 9;

  std::cout << m1 << std::endl;
  

  std::cout << "\n m4 before : \n" << m4 << std::endl;
  
  m4(1,1) = m4(0,0) + m4(0,1);
  
  std::cout << "\n m4 after : \n" << m4 << std::endl;
}

/*
0 3 0
0 0 0
0 0 9


 m4 before :
 0.823295 -0.329554
-0.604897  0.536459

 m4 after :
 0.823295 -0.329554
-0.604897   0.49374
*/
```

<br>

추가적인 사용방법은 아래 표와 같다.

<img src="/assets/img/dev/week17/basic_matrix.png">

<br>

matrix나 vector에 값을 입력할 때는 **comma initializer**(`<<`)를 사용하면 편하다.

```cpp
int main()
{
    Eigen::Matrix3d intrinsic_matrix = Eigen::Matrix3d::Zero();

    intrinsic_matrix << 422.037858, 0.0,        245.895397,
                        0.0,        435.589734, 163.625535,
                        0.0,        0.0,        1.0;

    std::cout << intrinsic_matrix << std::endl;
}
/*  
422.037858 0.0        245.895397
0.0        435.589734 163.625535
0.0        0.0        1.0
*/
```

추가적인 것들은 공식 문서를 참고하길 바란다.

<br>

<br>

## Eigen - Linear Algebra

- 참고 자료 : [Eigen 공식 문서 linear algebra](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html)
- 참고 자료 : [Eigen 공식 문서 Solving linear least squares systems](https://eigen.tuxfamily.org/dox/group__LeastSquares.html)

<br>

<details open>
 <summary> linear algebra </summary>

- linear solver
    - [https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html)
- SVD
    - [https://eigen.tuxfamily.org/dox/group__LeastSquares.html](https://eigen.tuxfamily.org/dox/group__LeastSquares.html)
- sparse matrix
    - [https://eigen.tuxfamily.org/dox/group__TutorialSparse.html](https://eigen.tuxfamily.org/dox/group__TutorialSparse.html)
    - [https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html](https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html)
    - [https://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html](https://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html)

</details>

<br>

```cpp
//
// Created by dkssu on 2022-06-12.
//
// reference : https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html

#include <iostream>
#include <Eigen/Dense>

void linear_solve()
{
    // 1. Ax = b ( ColPivHouseholderQr solver )
    // refence : https://eigen.tuxfamily.org/dox/classEigen_1_1ColPivHouseholderQR.html
    // performs rank-revealing QR decomposition of matrix A, permutation matrix P, unitary matrix Q, upper triangular matrix R
    // A P = Q R
    // this decomposition performs column pivoting in order to be rank-revealing
    Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
    Eigen::Vector3f b = Eigen::Vector3f::Zero();

    A << 1,2,3,
         4,5,6,
         7,8,10;
    b << 3, 3, 4;

    Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
    std::cout << "colpivhouseholderQr solution is : \n" << x << std::endl;

    // we can predefine decomposition constructor size
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(50,50);
    Eigen::MatrixXf p = Eigen::MatrixXf::Random(50, 50);
    qr.compute(p); // no dynamic memory allocation



    // 2. Ax = b ( LDLT )
    Eigen::Matrix2f A2, b2;
    A2 << 2, -1,
          -1, 3;
    b2 << 1, 2,
          3, 1;

    Eigen::Matrix2f x2 = A2.ldlt().solve(b2);
    std::cout << "LDLT solution is : \n" << x2 << std::endl;
};



/*
 this is describes how to solve linear least squares systems using Eigen. define Ax = b, has no solution.
 it makes to search for the vector x which is closest to being a solution in the difference Ax - b is as small as possible.
 if Euclidean norm is used, x is called the least square solution.
 */
void least_squares()
{
    // Ax = b ( normal equations - A_T * A * x = A^T * b )
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
    Eigen::VectorXf b = Eigen::VectorXf::Random(3);
    Eigen::VectorXf x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    std::cout << "The solution using normal equations is : \n" << x << std::endl;

    /*
     ldlt perform a robust 'Cholesky decomposition' of positive or negative matrix A such that A = P^T * L * D * L * P
     where, permutation matrix P, lower triangular matrix(하삼각행렬) with unit diagonal L, diagonal matrix D

     cholesky decomposition are not rank-revealing.

     because using ldlt() instead of LU is lower computation. L^T is upper triangular matrix(상삼각행렬), so U convert to DL^T in 'A = LU'
     therefore, exists lower triangular matrix L, symmetric matrix D for symmetric matrix A
     reference : https://freshrimpsushi.github.io/posts/ldu-decomposition-of-symmetric-matrix/
     */




    // Ax = b ( SVD decomposition - BDCSVD )
    Eigen::MatrixXf svd_A = Eigen::MatrixXf::Random(3, 2);
    Eigen::VectorXf svd_b = Eigen::VectorXf::Random(3);
    std::cout << "Here is the matrix A : \n" << svd_A << "\n"
        << "Here is the right side vector b : \n" << svd_b << std::endl;
    std::cout << "The least squares solution is : \n"
        << svd_A.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(svd_b) << std::endl;





    // Ax = b ( QR decomposition )
    Eigen::MatrixXf qr_A = Eigen::MatrixXf::Random(3, 2);
    Eigen::MatrixXf qr_b = Eigen::VectorXf::Random(3);
    Eigen::MatrixXf qr_x = qr_A.colPivHouseholderQr().solve(qr_b);
    std::cout << "The solution using the QR decomposition is : \n" << qr_x << std::endl;




    // Ax = b ( fullPivLu ) -> check if matrix is singular
    Eigen::MatrixXd fullLu_A = Eigen::MatrixXd::Random(100, 100);
    Eigen::MatrixXd fullLu_b = Eigen::MatrixXd::Random(100, 50);
    Eigen::MatrixXd fullLu_x = fullLu_A.fullPivLu().solve(fullLu_b);
    double relative_error = ( fullLu_A * fullLu_x - fullLu_b ).norm() / fullLu_b.norm(); // norm is L2 norm
    std::cout << "the relative error is : \n" << relative_error << std::endl;




    // Ax = b ( LLT ) -> separate computation , all decompositions have a compute(matrix) method that does the computation, and that can called again on an already-computed decomposition
    Eigen::Matrix2f llt_A, llt_b;
    Eigen::LLT<Eigen::Matrix2f> llt; // predefine size to the decomposition constructor
    llt_A << 2, -1,
             -1, 3;
    llt_b << 1, 2,
             3, 1;

    // computing LLT decomposition
    llt.compute(llt_A);
    std::cout << "The solution is:\n" << llt.solve(llt_b) << std::endl;

    // +1 at (1,1) in A3 matrix
    llt_A(1,1)++;
    llt.compute(llt_A);
    std::cout << "The solution is:\n" << llt.solve(llt_b) << std::endl;
};

```

<br>

