# Fingerprint recognition

This repository is intended only to support my ending semester for Biometrics systems course on Msc. in Cybersecurity // University of Bari, Italy. 

Credits goes to authors: kjanko (Kristijan Jankoski), ogabriel (Gabriel Oliveira), sashuu69 (Sashwat K), Utkarsh Deshmukh (https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python), R. Cappelli, M. Ferrara, A. Franco and D. Maltoni (FVC2006)  

* Works through Python, SKimage and OpenCV
    * First of all minutiae points are extracted using harris corner detection
* Uses SIFT (ORB) go get formal descriptors around the keypoints with brute-force hamming distance
    * Analyzes the returned matches using thresholds

## Dependencies

* Python 3+
* Numpy (numpy)
* scikit Image (scikit-image)
* OpenCV2 (opencv-python)

## Examples

* Used for almost 3000 different samples of FVC2006: the Fourth International Fingerprint Verification Competition. Just set both proper directory in the loop and the master sample
* Otherwise you can place two fingerprint images that you want to compare inside the database folder

## Documentation

* Full documentation of my project in Biometric systems course is provided only on Italian (manual folder).

_For any questions or doubts, feel free to contact me at gabriele.patta@outlook.com_