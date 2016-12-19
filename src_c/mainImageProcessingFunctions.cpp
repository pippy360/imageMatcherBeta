#pragma once
#include <vector>
#include "opencv.hpp"
#include <stdio.h>

#include "FragmentHash.h"
#include "ShapeAndPositionInvariantImage.h"
#include "Triangle.h"

#define NUM_OF_ROTATIONS 1
#define TARGET_TRIANGLE_SCALE 200 //the fragments are scaled by this value 
#define HASH_SIZE 8

const std::vector<Keypoint> getTargetTriangle()
{
    std::vector<Keypoint> v;
    v.push_back(Keypoint(0,0));
    v.push_back(Keypoint(.5,0.83666003));//sqrt(0.7) == 0.83666003
    v.push_back(Keypoint(1,0));
    return v;
}

namespace cv
{


Matx33d calcTransformationMatrix(const std::vector<Keypoint>& inputTriangle, const std::vector<Keypoint>& targetTriangle)
{
	/*
	 * ######CODE BY ROSCA#######
	import numpy as np
	#T       A         B
	#[a b]   [pt1.x pt2.x]   [0.5 1]
	#[c d] x [pt1.y pt2.y] = [sin(.7) 0]

	#T*A = B
	#T*A*A^-1 = B*A^-1
	#TI = B*A^-1

	#TODO: fix scaling!
	areaOfTargetTriangle = 0.418330015
	#scale = np.sqrt( (area/areaOfTargetTriangle) )
	scale = 200
	scaleMat = np.matrix(((scale, 0), (0, scale)))

	A = np.matrix(np.concatenate((pt1, pt2), axis=1))
	B = np.matrix(((0.5, 1), (0.83666003, 0)))#np.sqrt(0.7) == 0.83666003
	B = scaleMat * B
	T = B * A.getI()

	return T
	 */	
	Keypoint target_pt1 = targetTriangle[1];
	Keypoint target_pt2 = targetTriangle[2];
	Keypoint input_pt1 = inputTriangle[1];
	Keypoint input_pt2 = inputTriangle[2];
	cv::Matx33d scale_m(  TARGET_TRIANGLE_SCALE, 0.0, 0.0,
						   0.0, TARGET_TRIANGLE_SCALE, 0.0,
						   0.0, 0.0, 1.0 );

	cv::Matx33d targetPoints(  target_pt1.x, target_pt2.x, 0.0,
							   target_pt1.y, target_pt2.y, 0.0,
							   0.0, 0.0, 1.0 );

	cv::Matx33d inputPoints(  input_pt1.x, input_pt2.x, 0.0,
							  input_pt1.y, input_pt2.y, 0.0,
							  0.0, 0.0, 1.0 );

	//return targetPoints * (scale_m*inputPoints.inv());
	return  (targetPoints * (scale_m*inputPoints.inv()));
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
    auto pt2_t = Keypoint(pt2.x-pt1.x, pt2.y-pt1.y);
    auto pt3_t = Keypoint(pt3.x-pt1.x, pt3.y-pt1.y);

	auto ret = std::vector<Keypoint>();
	ret.push_back(Keypoint(0,0));
    if( isToTheLeftOf(pt2_t, pt3_t) ){
		ret.push_back(pt2_t);
		ret.push_back(pt3_t);
	} else {
		ret.push_back(pt3_t);
		ret.push_back(pt2_t);
	}
	return ret;			
}

//@shift: shift the points this many positions, the last point rolls over to the first position each shift
//@shift: this is used to get every rotation of the triangle we need (3, one for each edge)
const std::vector<Keypoint> prepShapeForCalcOfTransformationMatrixWithShift(const std::vector<Keypoint> shape, const std::vector<Keypoint>& targetTriangle, unsigned int shift = 0)
{
	auto t = prepShapeForCalcOfTransformationMatrix(shape, targetTriangle);
	shift %= t.size();
	//printf("this is the shift: %d\n", shift);
	std::rotate(t.begin(),t.begin()+shift,t.end());
	return t;
}

Mat applyTransformationMatrixToImage(Mat inputImage, const Matx33d transformation_matrix)
{
	//strip the bottom row
	
	Mat outputImage(8, 8, CV_8UC3, Scalar(0,0,0));
	//Mat outputImage(800, 900, CV_8UC3, Scalar(0,0,0));
	//auto outputImage = inputImage.clone();
	//Mat outputImage(200*.83, 200, CV_8UC3, Scalar(0,0,0));
	cv::Mat m = cv::Mat::ones(2, 3, CV_64F);
	m.at<double>(0, 0) = transformation_matrix(0, 0);
	m.at<double>(0, 1) = transformation_matrix(0, 1);
	m.at<double>(0, 2) = transformation_matrix(0, 2);
	m.at<double>(1, 0) = transformation_matrix(1, 0);
	m.at<double>(1, 1) = transformation_matrix(1, 1);
	m.at<double>(1, 2) = transformation_matrix(1, 2);
	//std::cout << m << std::endl;
	cv::warpAffine(inputImage, outputImage, m, outputImage.size());
	//imshow("out", outputImage);
	//waitKey();
//	m = getRotationMatrix2D(Point2f(0,0), 10, 1);
	m.at<double>(0, 0) = 1;
	m.at<double>(0, 1) = 0;
	m.at<double>(0, 2) = 100;
	m.at<double>(1, 0) = 0;
	m.at<double>(1, 1) = 1;
	m.at<double>(1, 2) = 100;

	//std::cout << m << std::endl;
	//cv::warpAffine(inputImage, outputImage, m, outputImage.size());
	auto s = outputImage.size();
	return outputImage;
}

std::vector<ShapeAndPositionInvariantImage> normaliseScaleAndRotationForSingleFrag(ShapeAndPositionInvariantImage& fragment)
{
	auto shape = fragment.getShape();

	auto ret = std::vector<ShapeAndPositionInvariantImage>();
	for (int i = 0; i < NUM_OF_ROTATIONS; i++)
	{	
	
		for (auto s : shape)
		{
			//printf("old shape (%.2lf, %.2lf)\n", s.x, s.y);
		}

		auto newShape = prepShapeForCalcOfTransformationMatrix(shape, getTargetTriangle());
		//printf("for a new shape\n");

		for (auto s : newShape)
		{
			//printf("new shape (%.2lf, %.2lf)\n", s.x, s.y);
		}
		 

		auto transformationMatrix = calcTransformationMatrix(newShape, getTargetTriangle());
		cv::Matx33d transpose_m(  1.0, 0.0, -shape[0].x,
								0.0, 1.0, -shape[0].y,
								0.0, 0.0, 1.0 );
		transformationMatrix = transformationMatrix * transpose_m;
		auto input_img = fragment.getImageData();
		//std::cout << transformationMatrix << std::endl;
		//imshow("d", fragment.getImageData());
		//waitKey();
		cv::Scalar scl = Scalar(0, 0, 255);

		cv::line(input_img, Point2f(shape[2].x, shape[2].y), Point2f(shape[0].x, shape[0].y), scl);
		cv::line(input_img, Point2f(shape[0].x, shape[0].y), Point2f(shape[1].x, shape[1].y), scl);
		cv::line(input_img, Point2f(shape[1].x, shape[1].y), Point2f(shape[2].x, shape[2].y), scl);
		auto newImageData = applyTransformationMatrixToImage(input_img, transformationMatrix);
		//imshow("here", newImageData);
		//waitKey();
		//imwrite("testFrag.jpg", newImageData);
		//imshow("d", newImageData);
		//waitKey();
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

std::vector<bool> dHashSlowWithResizeOrGrayscale(const Mat input_mat)
{
	Mat gray_image;
	cvtColor(input_mat, gray_image, CV_BGR2GRAY);
	
	int height = HASH_SIZE;
	int width = HASH_SIZE+1;
	Mat resized_input_mat;
	resize(gray_image, resized_input_mat, cvSize(width, height));

	return dHashSlowWithoutResizeOrGrayscale(resized_input_mat);
}

FragmentHash getHash(ShapeAndPositionInvariantImage frag)
{
	auto hash = dHashSlowWithResizeOrGrayscale(frag.getImageData());
	return FragmentHash(hash);
}

std::vector<FragmentHash> getHashesForFragments(std::vector<ShapeAndPositionInvariantImage>& normalisedFragments)
{
	auto ret = std::vector<FragmentHash>();//normalisedFragments.size()
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
//	for (auto tri : triangles)
//	{
	for (int i = 0; i < triangles.size(); i++)
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

}


