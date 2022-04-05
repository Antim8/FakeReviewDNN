""" This is the main script for user interaction with the program
"""
from model import Fake_detection


def main():
    
    print("Hello there fellow human! Welcome to the Fake Review Detector!")
    print("Please put your reviews into the \"ToAnalyze\" folder according to the Readme.md file in it")
    print("Once you are done, please hit a key to continue")
    input()
    print("Analyzing...")
    
    revs = []
    
    with open ("./ToAnalyze/revs.txt", "r") as f:
        for l in f:
            revs.append(str(l))
            
    print("Reviews read")
    
    fd = Fake_detection(classifier=True)
    encoded_revs = fd.encode(revs)
    sols = []
    
    for rev in encoded_revs:
        sols.append(fd(rev))
    
    print("Analyzing done")
    
    with open ("./ToAnalyze/results.txt", "W") as f:
        for i in range(len(revs)):
            f.write(f"{revs[i]}" + " isFake: " + str(sols[i]) + "\n")
            
    print("The results are in the \"ToAnalyze/result.txt\" file")


if __name__ == "__main__":
    main()