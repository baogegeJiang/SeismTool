import shapefile
import sys

def convert(inFile,outFile):
    with open(outFile,'w+') as outF:
        with shapefile.Reader(inFile) as inF:
            for shape in inF.shapes():
                outF.write('>\n')
                for point in shape.points:
                    outF.write('%f %f\n'%(point[0],point[1]))

if __name__ == "__main__":
    convert(sys.argv[1],sys.argv[2])
