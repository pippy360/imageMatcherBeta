import imagehash as ih
from utils import basicShapeOperations as BSO
from utils import basicImageOperations as BIO
import fragProcessing as fp
from Fragment import NormalisedFragment, FragmentImageData
#DEBUG IMPORTS#
import cv2
import time
import sys
#\DEBUG IMPORTS#


def getTheKeyPoints(img):
	from KeypointSystem import getKeypoints as gk
	k = gk.getTheKeyPoints(img)
	return k


def getTheTriangles(keyPoints, DEBUG_IMAGE=None, DEBUG_KEYPOINTS=None):
	from KeypointSystem import getKeypoints as gk
	return gk.getTheTriangles(keyPoints, DEBUG_IMAGE=DEBUG_IMAGE, DEBUG_KEYPOINTS=DEBUG_KEYPOINTS)


def buildNonNormalisedFragments(imgName, img, trianglesList):
	import fragProcessing as fs
	for triangle in trianglesList:
		#start = time.time()
		fragmentCoords, fragmentImage = fs.cutOutTheFrag(triangle, img)
		#end = time.time()
		##print "cut out the frag: " + str(end - start)
		nonNormFrag = FragmentImageData(fragmentImage, fragmentCoords)
		yield NormalisedFragment(imgName, triangle, None, nonNormFrag, None)


def _weNeedToAdd180(rot, shape):
	shape = BSO.rotateShapeAroundShapeCenter(shape, rot)
	shape = BSO.centerShapeUsingPoint(shape, (0,0))
	count = 0
	for pt in shape:
		if pt[1] < 0:
			count = count+1

	if count > 1:
		return True
	else: 
		return False


def getNormalisedFragmentForRotation(fragImageWithScaleFix, shapeWithScaleFix, rotation):
	if _weNeedToAdd180(rotation, shapeWithScaleFix):
		rotation = rotation + 180

	return BIO.rotateAndFitFragmentWhileKeepingShapeCenterAtTheCenterOfTheImage(fragImageWithScaleFix, rotation, shapeWithScaleFix)


def normaliseRotationForSingleFrag(inputImageData, inputShape):
	#returns the 3 rotation required to make the triangle sit on it's sides
	rotations = BSO.getFlatRotations(inputShape)

	fragmentImageDataList = []
	for rotation in rotations:
		rotatedShape, rotatedFrag = getNormalisedFragmentForRotation(inputImageData, inputShape, rotation)
		fragmentImageData = FragmentImageData(rotatedFrag, rotatedShape)
		fragmentImageDataList.append( fragmentImageData )

	return fragmentImageDataList


def _rotateAndScaleFragAndShape(inputFrag, angle, scalar):
	resShape, fragImageWithScaleFix = BIO.scaleImageAlongAxis_withCropping(inputFrag.fragmentImageShape, inputFrag.fragmentImage, angle, scalar) 
	return FragmentImageData(fragImageWithScaleFix, resShape)

def getPointClosestToZero(inputPts):
	minDist = BSO.dist((0,0), inputPts[0])
	minDistPt = inputPts[0]
	for pt in inputPts:
		tempDist = BSO.dist((0,0), pt)
		if tempDist < minDist:
			minDistPt = pt
			minDist = tempDist

	return minDistPt

#make sure the triangle won't be 'flipped'
def arrangePointsForTransformation(movedTri):
	ret = [(0,0),(0,0),(0,0)]
	temp_pt = getPointClosestToZero(movedTri)
	ret[0] = temp_pt
	#TODO: this might not be needed

def prepShapeForCalculationOfTranslationMatrix(fragmentImageShape):
	#get the point closest to zero
	from utils import basicShapeOperations as BSO
	import numpy as np

	tri = fragmentImageShape
	x_trans = tri[0][0]
	y_trans = tri[0][1]
	pt1 = (tri[1][0] - x_trans, tri[1][1] - y_trans)
	pt2 = (tri[2][0] - x_trans, tri[2][1] - y_trans)
	ang1 = BSO.getAngleBetweenTwoPoints(pt1, (0,0))
	ang2 = BSO.getAngleBetweenTwoPoints(pt2, (0,0))

	import math
	t1 = math.atan2(pt1[1], pt1[0])
	t1 %= 2*math.pi
	#print t1
	t2 = math.atan2(pt2[1], pt2[0])
	t2 %= 2*math.pi
	#print t2
	if t1 < t2:
		return np.matrix(pt1).T, np.matrix(pt2).T, -x_trans, -y_trans
	else:
		return np.matrix(pt2).T, np.matrix(pt1).T, -x_trans, -y_trans

def getTransformationMatrix(fragmentImageShape):
	import numpy as np
	b, c, x_translation, y_translation = prepShapeForCalculationOfTranslationMatrix(fragmentImageShape)
	area = BSO.getAreaOfTriangle(fragmentImageShape)
	translationMatrix = np.float32([
			[1,0,x_translation],
			[0,1,y_translation],
			[0,0,1]
		])

	t = calcTransformationMat(b, c, area)
	rotationAndScaleMatrix = np.float32([
		[t.item(0),t.item(1),0],
		[t.item(2),t.item(3),0],
		[0,0,1]
	])
	
	res = np.matmul(rotationAndScaleMatrix,translationMatrix)
	
	res = np.float32([
		[res.item(0),res.item(1),res.item(2)],
		[res.item(3),res.item(4),res.item(5)]
	])

	return res

#Code by Rosca
#No need to pass in point A because it MUST always be (0,0)
def calcTransformationMat(pt1, pt2, area):
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


def applyTransformationMatrixToFragment(inputImageData, transformationMatrix):
	import numpy as np

	imgFixed = cv2.warpAffine(inputImageData, transformationMatrix, (500,500))

	#now fix the shape

	t = transformationMatrix
	m = np.matrix(((t.item(0), t.item(1)),(t.item(3), t.item(4))))

	fixedShape = [(0,0), (100,200*0.83666003), (200,0)]

	return imgFixed, fixedShape


def normaliseScaleForSingleFrag(inputImageData, inputShape):
	transformationMatrix = getTransformationMatrix(inputShape)

	scaledImage, scaledShape = applyTransformationMatrixToFragment(inputImageData, transformationMatrix)

	return scaledImage, scaledShape


def fitTrianglesIntoImage(theThreeTrianglesAndShapes):
	ret = []
	for fragmentImageData in theThreeTrianglesAndShapes:
		fragImageWithScaleAndRotationFix, shapeWithScaleAndRotationFix = fragmentImageData.fragmentImage, fragmentImageData.fragmentImageShape
		fittedShape, fittedImage = fp.fitFragTightToImage(shapeWithScaleAndRotationFix, fragImageWithScaleAndRotationFix);
		#cv2.imshow('f', fittedImage)
		#cv2.waitKey()
		ret.append( FragmentImageData(fittedImage, fittedShape) )

	return ret


def normaliseFragmentScaleAndRotationAndHash(fragmentObj, hashProvider):
	inputImageData = fragmentObj.croppedFragment.fragmentImage
	inputShape = fragmentObj.croppedFragment.fragmentImageShape

	scaledImage, scaledShape = normaliseScaleForSingleFrag(inputImageData, inputShape)

	theThreeTrianglesAndShapes = normaliseRotationForSingleFrag(scaledImage, scaledShape)

	#Fit the fragments as best we can into the square images
	theThreeTrianglesAndShapes = fitTrianglesIntoImage(theThreeTrianglesAndShapes)

	ret = []
	for miniFrag in theThreeTrianglesAndShapes:
		imageName 			= fragmentObj.imageName
		fragmentImageCoords = fragmentObj.fragmentImageCoords
		fragmentHash 		= hashProvider.getHashPlain(miniFrag.fragmentImage)
		croppedFragment 	= fragmentObj.croppedFragment
		normalisedFragment	= miniFrag
		t = NormalisedFragment(imageName, fragmentImageCoords, fragmentHash, croppedFragment, normalisedFragment)
		ret.append(t)

	return t

#returns list of list of 3 ( [[o1,o2,o3],[o1,o2,o3],...], one for each different rotation) 
def finishBuildingFragments(inputFragmentsAndShapes, hashProvider):
	for fragment in inputFragmentsAndShapes:
		yield normaliseFragmentScaleAndRotationAndHash(fragment, hashProvider)


def getHashForSingleFragment(fragmentImageData):
	import hashProvider
	#start = time.time()
	hash = hashProvider.getHash(fragmentImageData)
	#end = time.time()
	##print "getHash: " + str(end - start)
	return hash

	
def buildFragmentObjects(imgName, imageData, triangles):
	#turn the triangles into fragments of the image
	nonNormalisedFragments = buildNonNormalisedFragments(imgName, imageData, triangles)

	#normalise the scale and fragments
	import hashProvider
	framgentObjsList = finishBuildingFragments(nonNormalisedFragments, hashProvider)

	return framgentObjsList

##################################################
#	public Fragment[] getAllTheHashesForImage
##################################################
#the main wrapper function for processing an image
#inputImage: opencv matrix for the image in colour
def getAllTheHashesForImage(shapeAndPositionInvariantImage):

	imageData = shapeAndPositionInvariantImage.imageData	
	#get the keyPoints
	keyPoints = getTheKeyPoints(imageData)

	#turn the keyPoints into triangles	
	triangles = getTheTriangles(keyPoints)

	framgentObjsList = buildFragmentObjects(shapeAndPositionInvariantImage.imageName, imageData, triangles)
	return framgentObjsList, len(triangles)

def buildImageFragmentsMapByHash(shapeAndPositionInvariantImage):
	frags, size = getAllTheHashesForImage(shapeAndPositionInvariantImage)
	ret = {}
	i = 0
	for frag in frags:
		i = i + 1
		ret[str(frag.fragmentHash)] = frag
		print 'finished: ' + str(i) + '/' + str(size)

	return ret