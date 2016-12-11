import imagehash as ih
from imagehash import ImageHash

def getHash(fragmentImage):
    return getHashPlain(fragmentImage.fragmentImage)

def getHashPlain(fragmentImage):
    from PIL import Image
    pythonImageObj = Image.fromarray(fragmentImage)
    return ih.dhash(pythonImageObj)

def strHashToHashObj(strHash):
    return hex_to_hash(strHash)

def hex_to_hash(hexstr, hash_size=8):
    import numpy
    """
    Convert a stored hash (hex, as retrieved from str(Imagehash))
    back to a Imagehash object.
    """

    l = []
    count = hash_size * (hash_size // 4)
    if len(hexstr) != count:
        emsg = 'Expected hex string size of {}.'
        raise ValueError(emsg.format(count))
    for i in range(count // 2):
        h = hexstr[i*2:i*2+2]
        v = int("0x" + h, 16)
        l.append([v & 2**i > 0 for i in range(8)])
    return ImageHash(numpy.array(l))

    