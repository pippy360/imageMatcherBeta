#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iomanip>      // std::setw
#include <math.h>       /* pow */

#include "FragmentHash.h"
#include "ShapeAndPositionInvariantImage.h"
#include "Triangle.h"

#define NUM_OF_ROTATIONS 3
#define HASH_SIZE 8
#define TARGET_TRIANGLE_SCALE 10 //the fragments are scaled by this value 
#define FRAGMENT_WIDTH TARGET_TRIANGLE_SCALE*HASH_SIZE
#define FRAGMENT_HEIGHT TARGET_TRIANGLE_SCALE*(HASH_SIZE+1)

const std::vector<Keypoint> getTargetTriangle()
{
    std::vector<Keypoint> v;
	//multiply all points by TARGET_TRIANGLE_SCALE
    v.push_back(Keypoint(0,0));
    v.push_back(Keypoint(.5*FRAGMENT_WIDTH,1*FRAGMENT_HEIGHT));//sqrt(0.7) == 0.83666003
    v.push_back(Keypoint(1*FRAGMENT_WIDTH,0));
    return v;
}

namespace cv
{


Matx33d calcTransformationMatrix(const std::vector<Keypoint>& inputTriangle, const std::vector<Keypoint>& targetTriangle)
{
	/*
	 * ######CODE BY ROSCA#######
	 */
	Keypoint target_pt1 = targetTriangle[1];
	Keypoint target_pt2 = targetTriangle[2];
	cv::Matx33d targetPoints(  target_pt1.x, target_pt2.x, 0.0,
							   target_pt1.y, target_pt2.y, 0.0,
							   0.0, 0.0, 1.0 );

	Keypoint pt2 = Keypoint(inputTriangle[1].x - inputTriangle[0].x, inputTriangle[1].y - inputTriangle[0].y);
	Keypoint pt3 = Keypoint(inputTriangle[2].x - inputTriangle[0].x, inputTriangle[2].y - inputTriangle[0].y);
	cv::Matx33d inputPoints(  pt2.x, pt3.x, 0.0,
							  pt2.y, pt3.y, 0.0,
							  0.0, 0.0, 1.0 );

	cv::Matx33d transpose_m(    1.0, 0.0, -inputTriangle[0].x,
								0.0, 1.0, -inputTriangle[0].y,
								0.0, 0.0, 1.0 );

	return  targetPoints * inputPoints.inv() * transpose_m;
}

bool isToTheLeftOf(Keypoint pt1, Keypoint pt2)
{
    return ((0 - pt1.x)*(pt2.y - pt1.y) - (0 - pt1.y)*(pt2.x - pt1.x)) > 0;
}

const std::vector<Keypoint> prepShapeForCalcOfTransformationMatrix(const std::vector<Keypoint>& inputTriangle, const std::vector<Keypoint>& targetTriangle)
{
	auto pt1 = inputTriangle[0];
	auto pt2 = inputTriangle[1];
	auto pt3 = inputTriangle[2];

	auto ret = std::vector<Keypoint>();
	ret.push_back(pt1);
    if( isToTheLeftOf(pt2, pt3) ){
		ret.push_back(pt2);
		ret.push_back(pt3);
	} else {
		ret.push_back(pt3);
		ret.push_back(pt2);
	}
	return ret;			
}

//@shift: this is used to get every rotation of the triangle we need (3, one for each edge of the triangle)
const std::vector<Keypoint> prepShapeForCalcOfTransformationMatrixWithShift(const std::vector<Keypoint> shape, const std::vector<Keypoint>& targetTriangle, int shift)
{
	auto shape_cpy = shape;
	shift %= shape_cpy.size();
	std::rotate(shape_cpy.begin(),shape_cpy.begin()+shift,shape_cpy.end());
	//printf("this is the shift: %d\n", shift);
	return prepShapeForCalcOfTransformationMatrix(shape_cpy, targetTriangle);
}

Mat formatTransformationMat(const Matx33d transformation_matrix)
{
	cv::Mat m = cv::Mat::ones(2, 3, CV_64F);
	m.at<double>(0, 0) = transformation_matrix(0, 0);
	m.at<double>(0, 1) = transformation_matrix(0, 1);
	m.at<double>(0, 2) = transformation_matrix(0, 2);
	m.at<double>(1, 0) = transformation_matrix(1, 0);
	m.at<double>(1, 1) = transformation_matrix(1, 1);
	m.at<double>(1, 2) = transformation_matrix(1, 2);
	return m;
}

Mat applyTransformationMatrixToImage(Mat inputImage, const Matx33d transformation_matrix)
{
	Mat m = formatTransformationMat(transformation_matrix);
	
	Mat outputImage(FRAGMENT_HEIGHT, FRAGMENT_WIDTH, CV_8UC3, Scalar(0,0,0));
	//Mat outputImage(200*.83, 200, CV_8UC3, Scalar(0,0,0));
	warpAffine(inputImage, outputImage, m, outputImage.size());
	//DEBUG
	imshow("fragmentAfterTransformation", outputImage);
	waitKey();
	//DEBUG
	return outputImage;
}

void drawLines(Mat input_img, vector<Keypoint> shape){
	cv::Scalar scl = Scalar(0, 0, 255);
	cv::line(input_img, Point2f(shape[2].x, shape[2].y), Point2f(shape[0].x, shape[0].y), scl);
	cv::line(input_img, Point2f(shape[0].x, shape[0].y), Point2f(shape[1].x, shape[1].y), scl);
	cv::line(input_img, Point2f(shape[1].x, shape[1].y), Point2f(shape[2].x, shape[2].y), scl);
}

Matx33d calcTransformationMatrixWithShapePreperation(const std::vector<Keypoint>& inputTriangle, const std::vector<Keypoint>& targetTriangle, int shift)
{
	auto newShape = prepShapeForCalcOfTransformationMatrixWithShift(inputTriangle, targetTriangle, shift);
	return calcTransformationMatrix(newShape, targetTriangle);
}

std::vector<ShapeAndPositionInvariantImage> normaliseScaleAndRotationForSingleFrag(ShapeAndPositionInvariantImage& fragment)
{
	auto shape = fragment.getShape();
	auto ret = std::vector<ShapeAndPositionInvariantImage>();
	for (int i = 0; i < NUM_OF_ROTATIONS; i++)
	{	
		auto transformationMatrix = calcTransformationMatrixWithShapePreperation(shape, getTargetTriangle(), i);
		auto input_img = fragment.getImageData();
		//DEBUG
		//drawLines(input_img, shape);
		//DEBUG
		auto newImageData = applyTransformationMatrixToImage(input_img, transformationMatrix);
		auto t = ShapeAndPositionInvariantImage(fragment.getImageName(), newImageData, getTargetTriangle(), fragment.getImageFullPath());
		ret.push_back(t);
	}
	
	return ret;
}

ShapeAndPositionInvariantImage getFragment(const ShapeAndPositionInvariantImage& input_image, const Triangle& tri)
{
	//TODO: cut out the fragment
	auto ret = ShapeAndPositionInvariantImage("some frag", input_image.getImageData(), tri.toKeypoints(), "");
	return ret;
}

std::vector<bool> dHashSlowWithoutResizeOrGrayscale(Mat resized_input_mat)
{
	std::vector<bool> output;
	int width_j = resized_input_mat.cols;
	int height_i = resized_input_mat.rows;
	for (int i = 0; i < height_i; i++)
	{
		for (int j = 0; j < width_j; j++)// "width_j-1" skip the last run
		{
			if(j == width_j -1 ){
				continue;
			}
			unsigned char left = resized_input_mat.at<unsigned char>(i, j, 0);
			unsigned char right = resized_input_mat.at<unsigned char>(i, j+1, 0);
			output.push_back( right > left );
		}
	}
	return output;
}

std::vector<bool> dHashSlowWithResizeAndGrayscale(const Mat input_mat)
{
	Mat gray_image;
	cvtColor(input_mat, gray_image, CV_BGR2GRAY);
	
	int height = HASH_SIZE;
	int width = HASH_SIZE+1;
	Mat resized_input_mat;
	resize(gray_image, resized_input_mat, cvSize(width, height));

	return dHashSlowWithoutResizeOrGrayscale(resized_input_mat);
}

//returns hamming distance
int getHashDistance(FragmentHash first, FragmentHash second){
	auto hash1 = first.getHash();
	auto hash2 = second.getHash();
	assert(hash1.size() == hash2.size());

	int dist = 0;
	for (int i = 0; i < hash1.size(); i++)
	{
		dist += (hash1[i] != hash2[i])? 1:0;
	}
	return dist;
}

FragmentHash getHash(ShapeAndPositionInvariantImage frag)
{
	auto hash = dHashSlowWithResizeAndGrayscale(frag.getImageData());
	return FragmentHash(hash);
}

std::vector<FragmentHash> getHashesForFragments(std::vector<ShapeAndPositionInvariantImage>& normalisedFragments)
{
	auto ret = std::vector<FragmentHash>();
	for (auto frag : normalisedFragments)
	{
		ret.push_back(getHash(frag));
	}
	return ret;
}

std::vector<FragmentHash> getHashesForTriangle(ShapeAndPositionInvariantImage& input_image, const Triangle& tri)
{
	auto fragment = getFragment(input_image, tri);
	auto normalisedFragments = normaliseScaleAndRotationForSingleFrag(fragment);
	auto hashes = getHashesForFragments(normalisedFragments);

	return hashes;
}

std::vector<FragmentHash> getAllTheHashesForImage(ShapeAndPositionInvariantImage inputImage, std::vector<Triangle> triangles)
{
	auto ret = std::vector<FragmentHash>();//size==triangles.size()*NUM_OF_ROTATIONS
	// for (auto tri : triangles)
	// {
	for (int i = 0; i < 3; i++)
	{
		auto tri = triangles[i];
		auto hashes = getHashesForTriangle(inputImage, tri);
		for (auto hash: hashes)
		{
			ret.push_back(hash);
		}
	}
	return ret;
}

std::string convertHashToString(FragmentHash inHash)
{
	std::string ret = "";
	auto hash = inHash.getHash();
	int h = 0;
	for (int i = 0; i < hash.size(); i++)
	{
		if (hash[i]){
			h += pow(2, (i % 8));
		}

		if (i%8 == 7){
			std::stringstream buffer;
			buffer << std::hex << std::setfill('0') << std::setw(2) << h;
			ret += buffer.str();
			h = 0;
		}
	}
	return "hash: " + ret;
}

}


