

def getImageAndMatchingChangedImage(imgName, angleWereUsing = 45,  scaleWereUsing = 2):
    import cv2
    from utils import basicImageOperations as BIO
    originalImage = cv2.imread(imgName)
    print 'we just read the image... '+imgName
    im = originalImage
    originalImageShape = [(0,0), (im.shape[1],0), (im.shape[1], im.shape[0]), (0,im.shape[0])]

    #now make the next image    
    changedImage = originalImage.copy()
    im = changedImage
    changedImageShape = [(0,0), (im.shape[1],0), (im.shape[1], im.shape[0]), (0,im.shape[0])]
    changedImageShape, changedImage = BIO.rotateAndFitFragmentWhileKeepingShapeCenterAtTheCenterOfTheImage(changedImage, angleWereUsing, changedImageShape)
    changedImageShape, changedImage = BIO.scaleImageAlongAxis_withCropping(changedImageShape, changedImage, 0, scaleWereUsing)

    from ShapeAndPositionInvariantImage import ShapeAndPositionInvariantImage
    return (
        ShapeAndPositionInvariantImage(imgName+'_org.jpg', originalImage, originalImageShape), 
        ShapeAndPositionInvariantImage(imgName+'_changed.jpg', changedImage, changedImageShape)
    )

def buildTwoImagesWithMatchedTriangles(imgName):
    from TwoImagesWithMatchedTriangles import TwoImagesWithMatchedTriangles
    from TransformationObjects import Transformation
    from TwoImageKeypointSupplier import TwoImageKeypointSupplier
    angleWereUsing = 145
    scaleWereUsing = 2
    imageOrg, imageChanged = getImageAndMatchingChangedImage(imgName, angleWereUsing=angleWereUsing, scaleWereUsing=scaleWereUsing)
    trans = Transformation(scaleWereUsing, angleWereUsing, angleWereUsing, transpose=(0,0))
    kpSupplier = TwoImageKeypointSupplier(imageOrg, imageChanged, trans)
    ret = TwoImagesWithMatchedTriangles(imageOrg.imageData, imageChanged.imageData, trans, kpSupplier)
    return ret

def toSimpleTri(tri):
    ret = []
    for kpt in tri:
        ret.append(kpt.pt)
    return ret

def getTheTriCollection(tri, img):
    import mainImageProcessingFunctions as mp
    org_tri = [toSimpleTri(tri)]
    org_frag_list = mp.getTheFragments(img, org_tri)
    normFrag = mp.normaliseScaleAndRotationForAllFrags(org_frag_list)
    return normFrag

def getDiffHash(tri1, tri2):
    from hashProvider import getHash
    return getHash(tri1) - getHash(tri2)

def getTheMatchingOnesBasedOnHash(list1, list2):
    list2Popped = list(list2)
    ret = []
    for tri in list1:
        matchIdx = 0
        closest = getDiffHash(list2Popped[0], tri)
        for i in range(len(list2Popped)):
            diffHash = getDiffHash(list2Popped[i], tri)
            if diffHash < closest:
                matchIdx = i
                closest = diffHash 

        t = (tri, list2Popped[matchIdx])
        ret.append(t)
        #del list2Popped[matchIdx]
        
    return ret

def test():
    import cv2
    from hashProvider import getHash
    imgName = '../input/img1.jpg'
    twoImagesWithMatchedTriangles = buildTwoImagesWithMatchedTriangles(imgName)
    print '#########test#########'
    tris = twoImagesWithMatchedTriangles.getMatchingTriangles()
    for triObj in tris:
        #grab the actual frag from each
        #and assert that they match!!
        #"transformedImageTriangle"
        #"originalImageTriangle"

        triCollection_org = getTheTriCollection(triObj['originalImageTriangle'], twoImagesWithMatchedTriangles.originalImage)
        triCollection_trans = getTheTriCollection(triObj['transformedImageTriangle'], twoImagesWithMatchedTriangles.transformedImage)

        #NOW TEST THAT THEY MATCH 
        for list1, list2 in zip(triCollection_org, triCollection_trans):
            arrangedTris = getTheMatchingOnesBasedOnHash(list1, list2)
            for x in arrangedTris:
                im1 = x[0]
                im2 = x[1]
                print "Distance: " + str(getHash(im1) - getHash(im2)) + " Hash img1: " + str(getHash(im1)) + " - Hash img2: " + str(getHash(im2)) 
                cv2.imshow('1', im1.fragmentImage)
                cv2.imshow('2', im2.fragmentImage)
                cv2.waitKey()
test()