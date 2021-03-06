---
title:    "[데브코스] 5주차 - OpenCV primary Class "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-15 01:45:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, opencv]
toc: True
comments: True
# image:
#   src: /assets/img/dev/week5/day2/main.png
#   width: 800
#   height: 500
---

<br>

# OpenCV 주요 클래스

## Point_ 클래스

point_ 클래스 : 2차원 점의 좌표 표현을 위한 템플릿 클래스
- 멤버 변수 : x,y ( 점 좌표 )
- 멤버 함수 : 
  - dot() : 내적 계산
  - ddot() : 더블 타입으로 내적 계산
  - cross() : 외적 계산
  - inside() : 한 점이 어떤 사각형 안에 존재 여부
- 다양한 사칙 연산에 대한 연산자 오버로딩과 std::cout 출력을 위한 << 연산자(shift 연산자) 오버로딩을 지원

아래는 point_ 클래스의 정의를 코드화시킨 것이다. typedef 부분은 생성자 부분으로 어떻게 타입을 넣느냐에 따라 다르다는 것을 보여준다.

```cpp
template<typename _Tp> class Size_
{
  public:
    ...
    _Tp x, y; // the point coordinates
}
typedef Point_<int>     Point2i;
typedef Point_<int64>   Point2l;
typedef Point_<float>   Point2f;
typedef Point_<double>  Point2d;
typedef Point2i         Point;
```

가장 많이 사용되는 정수형태를 Point2i로 정의한 후 그것을 다시 Point로 정의하여 쉽게 사용할 수 있도록 해놓았다.

<br>

- Point 연산의 예

point 객체끼리의 덧셈과 뺄셈, 한 점과 숫자 값과의 곱셈 등의 연산을 지원한다.

```cpp
Point pt1, pt2(4,3), pt3(2,4)   // pt1 == [0, 0]
int a = 2;

pt1 = pt2 + pt3;                // pt1 == [6, 7]
pt1 = pt2 - pt3;                // pt1 == [2, -1]
pt1 = pt3 * a;                  // pt1 == [4, 8]
pt1 = a * pt3;                  // pt1 == [1, 2]
pt1 += pt2;                     // pt1 == [5, 5]
pt *= a;                        // pt1 == [2, 4]

double v = norm(pt2);           // v = 5.0  원점에서 점까지의 거리
bool b1 = pt1 == pt2;           // b1 = false
bool b2 = pt1 != pt2;           // b2 = true

cout << pt1 << endl;            // "[1, 2]"
```

<br>

## Size_ 클래스

size_ 클래스 : 영상 또는 사각형의 크기 표현을 위한 템플릿 클래스
- 멤버 변수 : width, height
- 멤버 함수 : area() - 사각형의 면적 계산
- 다양한 사칙 연산에 대한 연산자 오버로딩과 std::cout 출력을 위한 << 연산자 오버로딩을 지원

```cpp
template<typename _Tp> class Size_
{
  public:
    ...
    _Tp width, height; // the width and the height
};

typedef Size_<int>    Size2i;
typedef Size_<int64>  Size2l;
typedef Size_<float>  Size2f;
typedef Size_<double> Size2d;
typedef Size2i        Size;
```

<br>

## Rect_ 클래스

rect_ 클래스 : 2차원 사각형 표현을 위한 템플릿 클래스
- 멤버 변수 : x, y, width, height
- 멤버 함수 : 
  - tl() : 좌측 상단 점의 좌표를 반환
  - br() : 오른쪽 하단의 점의 좌표를 반환
  - size() : size 객체를 반환
  - area() : 사각형 크기 계산
  - contains() : 한 점이 어떤 사각형안에 포함되어 있는지 여부
- 다양한 사칙 연산에 대한 연산자 오버로딩과 std::cout 출력을 위한 << 연산자 오버로딩을 지원

```cpp
template<typename _Tp> class Rect_
{
  public:
    ...
    _Tp x, y, width, height; // coordinate of upper left point and width and height
};

typedef Rect_<int>    Rect2i;
typedef Rect_<float>  Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i        Rect;
```

<br>

- Rect 연산의 예

Rect 객체와 Size, Point 객체의 덧셈과 뺄셈, Rect 객체끼리의 논리 연산을 지원한다.

```cpp
Rect rc1;                         // rc1 = [0 x 0 from (0, 0)]
Rect rc2(10, 10, 60, 40);         // rc2 = [60 x 40 from (10, 10)]
Rect rc3 = rc1 + Size(50, 40);    // rc3 = [50 x 40 from (0, 0)] 가로세로가 더해짐
Rect rc4 = rc2 + Point(10, 10);   // rc4 = [60 x 40 from (20, 20)] 좌표가 더해짐
Rect rc5 = rc3 & rc4;             // rc5 = [30 x 20 from (20, 20)] 두 사각형이 겹치는 최대 크기의 사각형
Rect rc6 = rc3 | rc4;             // rc6 = [80 x 60 from (0, 0)] 두 사각형을 모두 포함할 수 있는 최소 크기의 사각형
```

<img src="/assets/img/dev/week5/day2/intersectionandcombination.png">

<br>

## Range 클래스

Range 클래스 : 정수값의 범위를 나타내기 위한 클래스
- 멤버 변수 : start, end
- 멤버 함수 : 
  - size() : 범위의 크기 계산
  - empty() : 해당 범위가 비어있는지에 대한 여부
  - all() : 범위 안의 모든 수
- start는 범위에 포함되고, end는 범위에 포함되지 않음 -\> [start, end)

```cpp
class Range
{
  public:
    Range();
    Range(int _start, int _end);
    int size() const;
    bool empty() const;
    static Range all();

    int start, end;
}
```

<br>

## String 클래스

string 클래스 : 원래는 OpenCV에서 자체적으로 정의해서 사요하는 문자열이 있엇으나 4.x 버전부터 std::string 클래스로 대체되었다.

```cpp
typedef std::string cv::String;
```

- cv::format() 함수를 이용하여 형식 있는 문자열 생성 가능 -\> c언어의 printf() 함수와 인자 전달 방식이 유사하다.

```cpp
String str1 = "Hello";
String str2 = std::string("world");
String str3 = str1 + " " + str2;

Mat imgs[3];
for (int i = 0; i < 3; i++) {
  String filename = format("test%02d.bmp", i+1); // format 함수의 형태로 생성이 가능하다. 이 때, d에 i+1이 들어가는데, 2자리수 형태로 들어가고, 앞의 자리수가 비어있으면 0으로 채우는 것 -=> test01.bmp
  imgs[i] = imread(filename);
}
```

<br>

## Vec 클래스

벡터(vector) : 같은 자료형 원소 여러 개로 구성된 데이터 형식 (열 벡터)
- vec클래스는 벡터로 표현하는 탬플릭 클래스다.
- std:cout 출력을 위해 << 연산자 오버로딩을 지원한다.
- Matx라는 클래스는 작은 크기의 행렬을 표현하는 클래스로 주로 16개 원소 이하에서 많이 사용한다.

```cpp
template<typename _Tp, int m, int n> class Matx
{
  public:
    ...
    _Tp val[m*n]; //  matrix elements
};

template<typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1> // 열의 크기를 1로 한정한 클래스
{
  public:
    ...
    const _Tp& operator [](int i) const;
    _Tp& operator[](int i);
}
```

- Vec 클래스 이름 재정의

자주 사용되는 자료형과 개수에 대한 vec 클래스 템플릿의 이름을 재정의해놓은 것이 있다.

```cpp
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;  // 컬러 영상의 픽셀값을 참조할 때 많이 사용한다.
typedef Vec<uchar, 4> Vec4b;

typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;
typedef Vec<int, 6> Vec6i;
typedef Vec<int, 8> Vec8i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;
```

<br>

## Scalar 클래스

scalar 클래스 : 크기가 4인 double 배열(double val[4])을 멤버 변수로 가지고 있는 클래스
- 4채널 이하의 영상에서 픽셀 값을 표현하는 용도로 자주 사용
- []연산자를 통해 원소에 접근 가능

```cpp
template<typename _Tp> class Scalar_ : public Vec<_Tp, 4> // scalar=원소 개수가 4개인 열벡터 (템플릿 클래스)
{
  public:
    Scalar_();
    Scalar_(_Tp v0, _Tp v1, _Tp v2=0, _Tp v3=0);
    Scalar_(_Tp v0);

    static Scalar_<_Tp> all(_Tp v0);
    ...
};
```

이렇게 정의되어 있지만 잘 사용하지 않는데 자료형을 double로 한정한 경우만 많이 사용한다.

```cpp
typedef Scalar_<double> Scalar; // 4개의 double형 데이터를 갖는 자료형 (val[0],va[1],val[2],val[3])
```

스칼라 클래스는 double 타입이라 효율이 좋다고는 못하지만, 코드작성의 편의성과 코드의 통일성을 위해 자주 사용된다.

- Scalar 클래스 객체 생성과 원소 참조 방법

원래는 Scalar 클래스는 double val[0:5]를 가지고 있는데 1개만 지정하는 경우 맨 앞, 즉 val[0] 에만 들어가게 된다. 3개를 지정해줄 경우 앞에서부터 val[0~2]가 지정되고 마지막은 0으로 들어간다. yello.val[i]로 출력하는 경우와 yellow[i]로 출력하는 경우가 같다.

<img src="/assets/img/dev/week5/day2/scalar.png">

<br>

## 행렬 기초

행렬(matrix) : 수나 기호, 수식 등을 네모꼴로 배열한 것, 괄호로 묶어서 표시한다.

<img src="/assets/img/dev/week5/day2/matrix.png">

이는 4행 4열로 된, 4x4 행렬이다. 행을 row, 열을 column이라 한다.

<img src="/assets/img/dev/week5/day2/rowcolvector.png">

이와 같이 열또는 행만 있는 것을 열벡터, 행벡터라고 한다. 이 두개를 통칭해서 벡터라 한다.

<br>

- 행렬의 덧/곱셈

행렬의 덧셈은 같은 자리의 숫자들끼리 더하지만 곱셈의 방식은 다르다. 

<img src="/assets/img/dev/week5/day2/matrixcalcul.png">

곱셈은 이와 같이 계산되기 때문에 `행의 수 == 열의 수` 또는 `열의 수 == 행의 수` 라는 조건이 붙는다. 예를 들어 [2x1] x [2x1] 이면 계산이 되지 않고, [1x2] x [2x1] 이면 계산이 된다.

<br>

- 역행렬

<img src="/assets/img/dev/week5/day2/inversematrix.jpg">

A x A^-1 = I, 이 때, I를 단위행렬이라 한다.

- 전치 행렬

<img src="/assets/img/dev/week5/day2/transposematrix.png">

<br>

<br>

## Mat 클래스

mat 클래스 : n차원 1채널 또는 다채널 행렬 표현을 위한 클래스다
- 실수 또는 복소수 행렬, grayscale 또는 truecolor 영상, 벡터 필드, 히스토그램, 텐서 등을 표현할 수 있다.
- 다양한 형태의 행렬 생성, 복사, 행렬 연산 기능을 제공한다.
  - 행렬 생성 시 행렬의 크기, 자료형의 채널 수(타입), 초기값등을 지정할 수잇다.
  - 복사 생성자 & 대입 연산자는 얕은 복사를 수행한다. 즉, 참조 계수로 관리한다.
  - 깊은 복사는 Mat::copyTo() 또는 Mat::clone() 함수를 사용한다.
  - 다양한 사칙 연산에 대한 연산자 오버로딩과 std::cout출력을 위한 << 연산자 오버로딩을 지원한다.
- 행렬의 원소(픽셀 값) 접근 방법을 제공한다.
  - Mat::data 멤버 변수가 실제 핅셀 데이터 위치를 가리킨다.
  - Mat::at<typename>(inty, int x) 또는 Mat::ptr<typename>(int y) 함수 사용을 권장한다.

<br>

```cpp
class Mat
{
  public:
    /* 생성자 & 소멸자 */
    Mat();
    Mat(int rows, int cols, int type);
    ...

    /* 멤버 함수 & 연산자 오버로딩 */
    vold create(int rows, int cols, int type);
    Mat& operator = (const Mat& m);
    Mat clone() const;
    void copyTo(OutputArray m) const;
    template<typename _Tp> _Tp* ptr(int i0 = 0);
    template<typename _Tp> _Tp& at(int row, int col);

    /* 멤버 변수 */
    int dims;       // 행렬의 차원, 영상 데이터의 경우 2차원으로 된다.
    int rows, cols; // rows = 행 = 세로, cols = 열 = 가로
    uchar* data;    // new 연산자를 이용해서 행렬의 원소 데이터의 메모리를 동적할당해서 메모리의 시작 주소를 저장할 포인터 변수 

    MatSize size;   // 영상의 크기를 정하는 것
    MatStep step;   // 영상 데이터 공간의 가로 크기를 정하는 것
}
```

### Mat 클래스의 깊이 (depth)

Mat 클래스의 깊이 : 행렬 원소가 사용하는 자료형 정보를 가리키는 매크로 상수(정보를 나타내는 용어)
- Mat::depth() 함수를 이용해서 참조한다.
- 형식 : `CV_ <bit-depth> {U|S|F}`

미리 정의된 매크로 상수에는 다음과 같다.

```cpp
#define CV_8U     0   // uchar, unsigned char
#define CV_8S     1   // schar, signed char 일반적인 char 타입
#define CV_16U    2   // ushort, unsigned short
#define CV_16S    3   // short  2byte signed 부호값
#define CV_32S    4   // int    일반적인 int
#define CV_32F    5   // float
#define CV_64F    6   // double
#define CV_16F    7   // float16_t
```

<br>

### Mat 클래스의 채널 (channel)

Mat 클래스의 채널 : 원소 하나가 몇 개의 채널로 구성되어 있는가에 대한 것이다.
- Mat::channels() 함수를 이용하여 참조한다.
- e.g. grayscale 영상은 픽셀 하나 당 밝기 값 1개, truecolor 영상은 픽셀 하나 당 B,G,R 색상 성분인 3개

<br>

### Mat 클래스의 타입 (type)

Mat 클래스의 타입 : 행렬의 깊이와 채널 수를 한꺼번에 나타내는 매크로 상수다.
- Mat::type() 함수를 이용하여 참조한다.
- 형식 : `CV_8UC1` -\> 앞부분에는 깊이 뒷부분에 채널을 붙인 것 
  - C1: 채널 수
  - U: 정수형 부호 , 정수형 부호(S/U), 실수형(F)
  - 8: 비트 수, 8/16/32/64

<br>

- Mat 클래스 속성 참조 예제

```cpp
Mat img = imread("lenna.bmp");

cout << "width: " << img.cols << endl;
cout << "Height: " << img.rows << endl;
cout << "channels: " << img.channels() << endl;

if (img.type() == CV_8UC1) 
  cout << "img is a grayscale image " << endl;
else if (img.type() == CV_8UC3) 
  cout << "img is a truecolor image " << endl;
```

<img src="/assets/img/dev/week5/day2/output.png">

<br>

### Mat 클래스 사용방법

- Mat 클래스 생성

Mat 클래스를 생성하는 방법에는 여러 가지가 있다

```cpp
Mat img1; // empty matrix

/* img 만들기 */
Mat img2(480, 640, CV_8UC1); // unsigned char, 1channel, (rows,cols, type)
Mat img3(480, 640, CV_8UC3); // unsigned char, 3channel
Mat img4(Size(640, 480), CV_8UC1); // Size(width, height), size로 할 때는 (가로, 세로)

Mat img5(480, 640, CV_8UC1, Scalar(128)); // unsigned char, 1channel, grayscale인 경우는 밝기 1개의 값만 필요함, (rows, cols, type, initial values)
Mat img6(480, 640, CV_8UC3, Scalar(0, 0, 255)); // unsigned char, 1channel, truecolor인 경우는 B,G,R 총 3개의 값이 필요함

/* mat클래스에서 제공하는 정적(static) 멤버 함수를 이용해서 행렬 객체를 생성할 수 있다. */
Mat mat1 = Mat::zeros(3, 3, CV_32SC1); // 0 matrix
Mat mat2 = Mat::ones(3, 3, CV_32SC1); // 1 matrix
Mat mat3 = Mat::eye(3, 3, CV_32SC1); // 단위 행렬 I, identity matrix

float data[] = {1, 2, 3, 4, 5, 6};
Mat mat4(2, 3, CV_32FC1, data); // data라는 배열을 참조하여 행렬 원소를 만들어라, (rows,cols, type, data)
/* 1 2 3
   4 5 6 */

data[0] = 100; // 배열을 참조하고 있다는 것을 주의해야 한다.
/* 100 2 3
     4 5 6 */

Mat mat5 = (Mat_<float>(2,3) << 1,2,3,4,5,6);   // opencv에서 제공하는 Mat_ 탬플릿 클래스와 shift 연산자를 사용하여 생성하는 방법
Mat mat6 = Mat_<uchar>({2,3}, {1,2,3,4,5,6}));  // 이 두가지 방법은 잘 사용하지 않는다.

mat4.create(256,256,CV_8UC3); // uchar, 3channels , 기존에 만들어져 있는 Mat객체의 데이터를 지우고 새로 입력하는 방법
mat5.create(4,4,CV_32FC1); // float, 1channel , (rows, cols, type), 초기값을 지정하는 것은 없기에 `=` 나 setTo로 입력해야 한다.

mat4 = Scalar(255,0,0); // 초기값, 즉 색상을 지정
mat5.setTo(1.f);        // 모든 요소를 1.f로 지정
```

<br>

- Mat클래스 객체의 참조와 복사

`=` 연산자는 참조, 즉 얕은 복사를 수행하고, clone이나 copyTo는 깊은 복사를 수행한다.

```cpp
Mat img1 = imread("dog.bmp");

Mat img2 = img1;  // 생성자 인자를 초기화하는 = 연산자, Mat img2(img1);, 얕은 복사
Mat img3;
img3 = img1;      // 대입하는 = 연산자, 얕은 복사

Mat img4 = img1.clone();  // clone은 img1의 복사본을 img4로 복사하는 것
Mat img5;
img1.copyTo(img5);        // img1의 데이터를 img5에 복사되는 것

img1.setTo(Scalar(0,255,255));  // yellow, img1만 바꿨을 때 1,2,3이 모두 바뀌는 것을 볼 수 있음

imshow("img1", img1);
imshow("img2", img2);
imshow("img3", img3);
imshow("img4", img4);
imshow("img5", img5);

waitKey();
destroyAllWindows();
```

<img src="/assets/img/dev/week5/day2/matop2be.png" cation="before">
<img src="/assets/img/dev/week5/day2/matop2af.png" cation="after">

간단하게 보면 img1 == img2를 하면 같은 메모리 공간을 참조하는 것이다. 그러나 img3 = img1.clone()을 하게되면 메모리 공간을 카피해서 메모리자체를 새로 생성하게 된다. 그래서 img1의 데이터를 변경하게 되면 img2는 바뀐 메모리의 데이터를 참조하기에 함께 바뀌지만, img3은 다른 메모리를 참조하기 때문에 변경되지 않는다.

<br>

- Mat 클래스 객체에서 부분 행렬 추출
  - Mat 객체에 대해 ()연산자를 이용하여 부분 영상 추출이 가능하다.
  - ()연산자 안에는 Rect 객체를 지정하여 부분 영상의 위치와 크기를 지정
  - 참조를 활용하여 ROI(Region of Interest) 연산 수행 가능

```cpp
Mat img1 = imread("cat.bmp");

Mat img2 = img1(Rect(220,120,340,240));         // (x,y,w,h), 저 사각형 크기로 잘라낸 데이터만 img2에 저장
Mat img3 = img1(Rect(220,120,340,240)).clone(); // == img2.clone() 저 사각형 크기로 된 부분을 깊은 복사

img2 = ~img2; // '~' 는 opencv에서 not 연산을 수행하는 연산자로, 이것을 반전시키는 방법이다.

imshow("img1",img1);
imshow("img2",img2);
imshow("img3",img3);

waitKey();
destroyAllWindows();
```

<img src="/assets/img/dev/week5/day2/matop3.png">

<br>

- 영상의 픽셀 값 참조

이는 openCV에서 제공하는 기능이 아닌 자신만의 새로운 기능을 추가할 때 많이 사용된다. 기본적으로 Mat::data 멤버 변수가 픽셀 데이터 메모리 공간을 가리키지만, Mat 클래스 멤버 함수를 사용하는 방법을 권장한다.

| 사용 함수 | 설명 |
| --- | --- |
| Mat::data | 메모리 연산이 잘못될 경우 프로그램이 비정상적으로 종료될 수 있다. 접근 식 : addr(Mi,j) = M.data + M.step[0] * i + M.step[1] * j, 이 때, i,j는 각각 행, 열번호에 해당하고 step[0]는 한 행의 byte크기, step[1]은 원소 하나의 byte크기 |
| Mat::at() | 좌표 지정이 직관적이다. 임의 좌표에 접근할 수 있다.|
| Mat::ptr() | Mat::at()보다 빠르게 동작한다. 행 단위 연산을 수행할 때 유리하다 |
| MatIterator_ | 좌표를 지정하지 않아서 안전하지만 성능이 느리다. |

<br>

- **Mat::at 함수 사용방법**

```cpp
template<typename _Tp> _Tp& Mat::at(int y, int x)
```

y: 참조할 행 번호

x: 참조할 열 번호

반환값 : (_Tp& 타입으로 캐스팅된) y행 x열 원소 값 참조

```cpp
Mat mat1 = Mat::zeros(3, 4, CV_8UC1);

for (int y = 0; y < mat1.rows; y++) {
  for (int x = 0; x < mat1.cols; x++) {
    mat1.at<uchar>(y, x)++; // at함수를 사용할 때 현재 mat클래스의 객체가 어떤 타입을 가지고 있는지 지정해야 한다. 이를 통해 참조 뿐만 아니라 설정도 가능하다.
    if (x==2 && y==1)
      mat1.at<uchar>(y, x)++; // 픽셀 값 +1
  }
}

cout << mat1 << endl;
```

<img src="/assets/img/dev/week5/day2/matop4.png">

<br>

- **Mat::ptr() 함수 사용 방법**

```cpp
template<typename _Tp> _Tp* Mat::ptr(int y)
```

y : 참조할 행 번호

반환값 : (_Tp* 타입으로 캐스팅된) y번 행의 시작 주소

```cpp
for (int y = 0; y < mat1.rows; y++) {
  uchar* p = mat1.ptr<uchar>(y); // 1개의 인자를 받게 되고, 데이터 타입을 지정해야 하고, 지정한 포인터를 반환한다.

  for (int x = 0; x < mat1.cols; x++) {
    p[x]++;
  }
}
```

uchar* p = mat1.ptr<uchar>(0) 을 하면 p는 mat1의 0번 행을 가리키게 된다. (1)이면 p는 mat1의 1번 행을 가리키게 될 것이다.

<br>

- **MatIterator_\<T\> 반복자 사용 방법**

openCV는 mat 클래스와 함께 사용할 수 있는 반복자 클래스 템플릿인 MatIterator_를 제공한다. MatIterator_는 클래스 템플릿이므로 사용할 때는 Mat 행렬 타입에 맞는 자료형을 명시해줘야 한다. Mat::begin() 함수는 행렬의 첫번째 원소 위치를 반환, Mat::end() 함수는 행렬의 **마지막 원소 바로 다음 위치를 반환**한다.

```cpp
for (MatIterator_<uchar> it = mat1.begin<uchar>(); it != mat1.end<uchar>(); ++it) {
		(*it)++;
	}
```

하지만 이 방법은 성능이 좋지 않아 잘 사용하지 않는다.

<br>

- 기초 행렬 연산

```cpp
float data[] = {1, 1, 2, 3};
Mat mat1(2, 2, CV_32FC1, data);     // 2x2 행렬의 원소는 data, type은 float32 1channel
cout << "mat1:\n" << mat1 << endl;  // 1 1;2 3

Mat mat2 = mat1.inv();              // 역행렬 함수
cout << "mat2:\n" << mat2 << endl;  // 3 -1;-2 1

cout << "mat1.t():\n" << mat1.t() << endl; // 전치행렬 함수
cout << "mat1 + 3:\n" << mat1 + 3 << endl; // 전체 +3
cout << "mat1 + mat2:\n" << mat1 + mat2 << endl; // 행렬 덧셈
cout << "mat1 * mat2:\n" << mat1 * mat2 << endl; // 행렬 곱셈
```

<img src="/assets/img/dev/week5/day2/matrixcal.png">

<br>

<br>

## InputArray 클래스

Inputarray 클래스 : 주로 Mat 클래스를 대체하는 프록시 클래스로 OpenCV 함수에서 입력 인자로 사용된다.

사용자가 명시적으로 _InuputArray 클래스의 인스턴스 또는 변수를 생성하여 사용하는 것은 금지되어 있다.

imshow나 imwrite를 할 시 2번째 인자의 경우 InputArray 클래스로 정의가 되어 있다. Mat이 아닌 inputarray인 이유는 mat이외에 Mat_<T>,Matx<T,m,n> 등 다양한 것들을 입력 받기 위해서다. 그러나 대체로는 **Mat**을 사용한다.

<br>

## OutputArray 클래스

outputarray 클래스 : opencv함수에서 출력 인자로 사용되는 프록시 클래스이다.

이 또한, Mat이 아닌 OutputArray인 이유도 다양한 타입을 받기 위해서이나, 대체로 Mat을 주로 사용한다.

<br>