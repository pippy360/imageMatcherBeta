#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <regex>

#include "FragmentHash.h"
#include "ShapeAndPositionInvariantImage.h"
#include "Triangle.h"
#include "mainImageProcessingFunctions.cpp"
#include <iostream>

void toTheLeftOfTest()
{

    //basic
    auto k1 = Keypoint(1,1);
    auto k2 = Keypoint(1,0);
    bool v = false;
    v = cv::isToTheLeftOf(k1, k2);
    if(v==true){
        printf("it worked!!\n");
    }else{
        printf("####ERROR: IT FAILED !!#######\n");
    }
    //TODO: more tests
}

void prepShapeForCalcOfTransformationMatrixTest()
{
    //basic
    auto k1 = Keypoint(0,0);
    auto k2 = Keypoint(1,0);
    auto k3 = Keypoint(1,1);
    auto v = std::vector<Keypoint>();
    v.push_back(k1);
    v.push_back(k2);
    v.push_back(k3);
    
    auto res = cv::prepShapeForCalcOfTransformationMatrix(v, getTargetTriangle());
    printf("{");
    for(auto t: res)
    {
        printf(" point: (%lf, %lf), ", t.x, t.y);
    }
    printf("}");    
    printf("\n");
}

void shiftTest()
{
    //basic
    auto k1 = Keypoint(0,0);
    auto k2 = Keypoint(1,0);
    auto k3 = Keypoint(1,1);
    auto v = std::vector<Keypoint>();
    v.push_back(k1);
    v.push_back(k2);
    v.push_back(k3);
    
    auto res = cv::prepShapeForCalcOfTransformationMatrixWithShift(v, getTargetTriangle(), 0);
    printf("{");
    for(auto t: res)
    {
        printf(" point: (%lf, %lf), ", t.x, t.y);
    }
    printf("}");
    printf("\n");

    res = cv::prepShapeForCalcOfTransformationMatrixWithShift(v, getTargetTriangle(), 1);
    printf("{");
    for(auto t: res)
    {
        printf(" point: (%lf, %lf), ", t.x, t.y);
    }
    printf("}");
    printf("\n");

    res = cv::prepShapeForCalcOfTransformationMatrixWithShift(v, getTargetTriangle(), 2);
    printf("{");
    for(auto t: res)
    {
        printf(" point: (%lf, %lf), ", t.x, t.y);
    }
    printf("}");
    printf("\n");

    //TODO: test shift > vec.size
}

void calcTransformationMatrixTest(){
    //basic
    auto k1 = Keypoint(0,0);
    auto k2 = Keypoint(1,0);
    auto k3 = Keypoint(1,1);
    auto v = std::vector<Keypoint>();
    v.push_back(k1);
    v.push_back(k2);
    v.push_back(k3);
    auto m = cv::calcTransformationMatrix(v, getTargetTriangle());
    //std::cout << "M = "<< std::endl << " "  << m << std::endl << std::endl;
}

void normaliseScaleAndRotationForSingleFragTest(cv::Mat &img){
	//cv::imshow("here", img);
	//cv::waitKey();
    auto k1 = Keypoint(0,0);
    auto k2 = Keypoint(100,0);
    auto k3 = Keypoint(100,100);
    auto v = std::vector<Keypoint>();
    v.push_back(k1);
    v.push_back(k2);
    v.push_back(k3);
    auto m = cv::calcTransformationMatrix(v, getTargetTriangle());
    //std::cout << "working so far M = "<< std::endl << " "  << m << std::endl << std::endl;    
    auto im = ShapeAndPositionInvariantImage("d", img, v, "something");
    cv::normaliseScaleAndRotationForSingleFrag(im);
}

void printTheDHash(std::vector<bool> hash){
    printf("\n");
    for(int i = 0; i < hash.size(); i++){
        if (i%8 == 0)
            printf("\n");
        printf("%s, ", (hash[i] == true)? "True":"False");
    }

    for(int i = 0; i < hash.size(); i++){
        printf("%s", (hash[i] == true)? "1":"0");
    }
    printf("\n");
}

cv::Mat prepImage(cv::Mat img)
{
    cv::Mat gray_image;
	cv::cvtColor(img, gray_image, CV_BGR2GRAY);
	
	int height = HASH_SIZE;
	int width = HASH_SIZE+1;

	cv::Mat resized_input_mat;
	resize(gray_image, resized_input_mat, cvSize(width, height));
	std::vector<bool> output;

	unsigned char temp[] = {
		124, 105, 130, 121, 129, 124, 255, 254, 255, 109, 100, 158, 171, 107, 129, 255, 253, 255, 111, 114, 103, 165, 105, 164, 255, 254, 255, 115, 81, 98, 132, 83, 177, 255, 254, 255, 107, 80, 102, 110, 106, 207, 255, 254, 255, 96, 62, 70, 131, 115, 158, 255, 255, 255, 137, 96, 128, 181, 150, 149, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
	};

	int width_j = resized_input_mat.cols;
	int height_i = resized_input_mat.rows;
	int count = 0;

	for (int i = 0; i < height_i; i++)
	{
		for (int j = 0; j < width_j; j++)// "width_j-1" skip the last run
		{
			//printf("%d: %d: %d: %d, ", temp[(i*width_j) + j], (i*width_j) + j, i ,j);
			resized_input_mat.at<unsigned char>(i, j, 0) = temp[(i*width_j) + j];
		}
	}

	printf("...\n, ");
    return resized_input_mat;
}

void dHashSlowTest(){
    cv::Mat img = cv::imread("./small_lenna1.jpg");
    cv::Mat fixed_img = prepImage(img);
    auto hash = cv::dHashSlowWithoutResizeOrGrayscale(fixed_img);
    bool tempBool[] = {
        0,1,0,1,0,1,0,1,0,1,1,0,1,1,
        0,1,1,0,1,0,1,1,0,1,0,1,1,0,
        1,1,0,1,0,1,1,0,1,1,0,1,0,1,
        1,0,1,1,0,0,0,1,1,0,0,1,0,1,
        0,0,0,0,0,0,0,0
        };
    int size = sizeof(tempBool) / sizeof(bool);
    printTheDHash(hash);
    if(size != hash.size()){
        printf("######ERROR: SIZE DOESN'T MATCH \n");
    }

    for (int i = 0; i < hash.size(); i++)
    {
        if(hash[i] != tempBool[i]){
            printf("######ERROR: HASH DOESN'T MATCH \n");            
        }else{
            //printf("Match\n");                        
        }
    }
    printf("done checking hashes\n");
}

void speedTest(){

    cv::Mat img = cv::imread("./small_lenna1.jpg");
    for(int i=0; i<1000;i++){
        normaliseScaleAndRotationForSingleFragTest(img);
    }
}

/*
std::vector<Keypoint>& getShapeFromImage(cv::Mat img)
{
	auto shape = img.size();
    return 
}
*/

const std::string readTheName(std::ifstream *file)
{
    std::string str;
    std::getline(*file, str);
    printf("file name is: %s\n", str.c_str());
    return str;
}

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

const std::vector<Triangle> readTheTriangles(std::ifstream *file)
{
    std::vector<Triangle> triangles;
    std::string str;
    while (true)
    {
        if(!std::getline(*file, str)){
            break;
        }
        
        double x1 = atof(str.c_str());
        std::getline(*file, str);
        double y1 = atof(str.c_str());
        Keypoint k1(x1, y1);

        std::getline(*file, str);
        double x2 = atof(str.c_str());
        std::getline(*file, str);
        double y2 = atof(str.c_str());
        Keypoint k2(x2, y2);

        std::getline(*file, str);
        double x3 = atof(str.c_str());
        std::getline(*file, str);
        double y3 = atof(str.c_str());
        Keypoint k3(x3, y3);

        Triangle t(k1, k2, k3);
        triangles.push_back(t);
    }
    return triangles;
}


void testHashConversion()
{
    std::vector<bool> hash;
    hash.push_back(false);
    hash.push_back(true);
    hash.push_back(false);
    hash.push_back(false);

    hash.push_back(false);
    hash.push_back(false);
    hash.push_back(false);
    hash.push_back(false);

    std::string str = cv::convertHashToString(FragmentHash(hash));
    std::cout << str << std::endl;
}

cv::Matx33d getATransformationMatrix(int width, int height){

    std::ifstream file("output.txt");
    std::string filename = readTheName(&file);
    auto tris = readTheTriangles(&file);
    auto tri = tris[0];
    
    return cv::calcTransformationMatrixWithShapePreperation(tri.toKeypoints(), getTargetTriangle(width, height), 0);
}

FragmentHash testHashingForResize()
{
    // cv::Mat img = cv::imread("./small_lenna1.jpg");
    cv::Mat img = cv::imread("../input/rick1.jpg");

    cv::Mat outputImage(200, 200, CV_8UC3, cv::Scalar(0,0,0));
    auto transformation_matrix = getATransformationMatrix(200, 200);
    // auto out2 = cv::applyTransformationMatrixToImage(img, transformationMatrix);

    cv::Mat m = formatTransformationMat(transformation_matrix);
    cv::warpAffine(img, outputImage, m, outputImage.size());

    auto hash = FragmentHash(cv::dHashSlowWithResizeAndGrayscale(outputImage));
    std::cout << cv::convertHashToString(hash) << std::endl;

	cv::imshow("here", outputImage);
	cv::waitKey();
    return hash;    
}

FragmentHash testHashingForResize2()
{
    // cv::Mat img = cv::imread("./small_lenna1.jpg");
    cv::Mat img = cv::imread("../input/rick1.jpg");

    cv::Mat outputImage(400, 400, CV_8UC3, cv::Scalar(0,0,0));
    auto transformation_matrix = getATransformationMatrix(400, 400);
    // auto out2 = cv::applyTransformationMatrixToImage(img, transformationMatrix);

    cv::Mat m = formatTransformationMat(transformation_matrix);
    cv::warpAffine(img, outputImage, m, outputImage.size());

    auto hash = FragmentHash(cv::dHashSlowWithResizeAndGrayscale(outputImage));
    std::cout << cv::convertHashToString(hash) << std::endl;

	cv::imshow("here", outputImage);
	cv::waitKey();
    return hash;    
}

FragmentHash testHashingForResize3()
{
    // cv::Mat img = cv::imread("./small_lenna1.jpg");
    cv::Mat img = cv::imread("../input/rick1.jpg");

    cv::Mat outputImage(8, 8+1, CV_8UC3, cv::Scalar(0,0,0));
    auto transformation_matrix = getATransformationMatrix(8+1, 8);
    // auto out2 = cv::applyTransformationMatrixToImage(img, transformationMatrix);

    cv::Mat m = formatTransformationMat(transformation_matrix);
    cv::warpAffine(img, outputImage, m, outputImage.size());

    auto hash = FragmentHash(cv::dHashSlowWithResizeAndGrayscale(outputImage));
    std::cout << cv::convertHashToString(hash) << std::endl;

	cv::imshow("here", outputImage);
	cv::waitKey();
    return hash;
}

FragmentHash testHashingForResize4()
{
    // cv::Mat img = cv::imread("./small_lenna1.jpg");
    cv::Mat img = cv::imread("../input/rick1.jpg");

    cv::Mat outputImage(500, 500, CV_8UC3, cv::Scalar(0,0,0));
    auto transformation_matrix = getATransformationMatrix(500, 500);
    // auto out2 = cv::applyTransformationMatrixToImage(img, transformationMatrix);

    cv::Mat m = formatTransformationMat(transformation_matrix);
    cv::warpAffine(img, outputImage, m, outputImage.size());

    cv::Mat resized_input_mat;
    int height = 8;
	int width = 8+1;
	resize(outputImage, resized_input_mat, cvSize(width, height));
    auto hash = FragmentHash(cv::dHashSlowWithResizeAndGrayscale(resized_input_mat));
    std::cout << cv::convertHashToString(hash) << std::endl;

	cv::imshow("here", outputImage);
	cv::waitKey();
    return hash;
}

int main(int argc, char* argv[])
{
	toTheLeftOfTest();
    prepShapeForCalcOfTransformationMatrixTest();
    shiftTest();
    calcTransformationMatrixTest();
    dHashSlowTest();
    testHashConversion();
    //speedTest();
    auto h1 = testHashingForResize();
    auto h2 = testHashingForResize2();
    auto h3 = testHashingForResize3();
    auto h4 = testHashingForResize4();

    int dist;
    dist = cv::getHashDistance(h1, h2);
    printf("dist1: %d\n", dist);
    dist = cv::getHashDistance(h1, h3);
    printf("dist2: %d\n", dist);
    dist = cv::getHashDistance(h2, h3);
    printf("dist3: %d\n", dist);
    dist = cv::getHashDistance(h1, h4);
    printf("dist4: %d\n", dist);
    dist = cv::getHashDistance(h2, h4);
    printf("dist5: %d\n", dist);
    dist = cv::getHashDistance(h3, h4);
    printf("dist6: %d\n", dist);

    // std::ifstream file("output.txt");
    // std::string filename = readTheName(&file);
    // auto tris = readTheTriangles(&file);

	// for (int i = 0; i < 10; i++)
	// {
	// 	//printf("new tri\n");
	// 	auto tri = tris[i];
	// 	auto k = tri.toKeypoints();
	// 	//printf("keypoint size: %d\n", k.size());
	// 	for (auto s : k)
	// 	{
	// 		//printf("(%.2lf, %.2lf)\n", s.x, s.y);
	// 	}
	// }

    // cv::Mat img = cv::imread("../input/rick1.jpg");
    // std::vector<Keypoint> g;
	// auto temp = ShapeAndPositionInvariantImage("small_lenna3", img, g, "");
	// auto k = cv::getAllTheHashesForImage(temp, tris);

    // printf("size: %d\n", k.size());
	// for(auto o: k){
    //     std::cout << cv::convertHashToString(o) << std::endl;
    // }

    // for(auto tri: tris)
    // {
	// 	/*
    //     printf("[(%.0lf, %.0lf), ", tri.keypoints_[0].x, tri.keypoints_[0].y);
    //     printf("(%.0lf, %.0lf), ", tri.keypoints_[1].x, tri.keypoints_[1].y);
    //     printf("(%.0lf, %.0lf)]\n", tri.keypoints_[2].x, tri.keypoints_[2].y);
	// 	 */
    // }
    // printf("done\n");
    /*
	cv::Mat img = cv::imread("../../small_lenna3.jpg");
	auto shape = getShapeFromImage(img);
	auto tris = cv::readTheTriangles();
	cv::imshow("here", img);
	cv::waitKey();
    */
	return 0;
}