//
//  main.cpp
//  OpenCVTester
//
//  Created by Michiro Hirai on 12/25/13.
//  Copyright (c) 2013 Sony. All rights reserved.
//

#include <array>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ocl/ocl.hpp>

class AlgorithmHolder {
private:

    std::array<std::string, 11> methods_ = {"SIFT", "FAST", "FASTX", "SURF", "STAR", "ORB", "BRISK", "MSER", "HARRIS", "Dense", "SimpleBlob" };

    std::vector<cv::Ptr<cv::FeatureDetector>> detectors_;
    std::vector<cv::KeyPoint> keypoints_;
    size_t current_;

public:
    AlgorithmHolder()
    : current_(0)
    {
        for (auto method : methods_) {
            detectors_.push_back(cv::FeatureDetector::create(method));
        }
        cv::initModule_nonfree();   // init for SIFT or SURF
    }

    std::string select(){

        if (++current_ >= methods_.size()) {
            current_ = 0;
        }
        return methods_.at(current_);
    }

    void detect(cv::Mat & image){
        cv::Mat result;
        detectors_.at(current_)->detect(image, keypoints_);
        cv::drawKeypoints( image, keypoints_, result);

        image = result;
    }

};

class VideoCaptureManager {
    cv::VideoCapture cap_;

public:
    VideoCaptureManager()
    : cap_(0)
    {
        if(!cap_.isOpened()){
            std::cerr << "failed to open" << std::endl;
        }
    }

    ~VideoCaptureManager()
    {
        cap_.release();
    }

    bool read( cv::Mat& image ) {
        return cap_.read(image);
    }

};

static void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    std::cerr << x << " : " << y << "event: " << event << std::endl;
}

static void draw(void * userdata)
{
//    cv::ogl::Texture2D * texture = static_cast<cv::ogl::Texture2D*>(userdata);
}

class FaceDetector {
    std::string face_cascade_name = "/Users/michiro/Dropbox/Development/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt.xml";
    std::string eyes_cascade_name = "/Users/michiro/Dropbox/Development/opencv-2.4.7/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    std::string smile_cascade_name = "/Users/michiro/Dropbox/Development/opencv-2.4.7/data/haarcascades/haarcascade_smile.xml";

    cv::CascadeClassifier face_cascade_;
    cv::CascadeClassifier smile_cascade_;
    cv::CascadeClassifier eyes_cascade_;
public:
    FaceDetector(){
        if(!face_cascade_.load(face_cascade_name)){
            std::cerr << "failed to load : " << face_cascade_name;
        }
        if(!smile_cascade_.load(smile_cascade_name)){
            std::cerr << "failed to load : " << smile_cascade_name;
        }
        if(!eyes_cascade_.load(eyes_cascade_name)){
            std::cerr << "failed to load : " << eyes_cascade_name;
        }
    }

    void detect( cv::Mat & frame);
};


void FaceDetector::detect( cv::Mat & frame )
{
    using namespace cv;

    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade_.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( auto face : faces )
    {
        Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
        ellipse( frame, center, Size( face.width*0.5, face.height*0.5), 0, 0, 360, Scalar( 0, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( face );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade_.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point center( face.x + eyes[j].x + eyes[j].width*0.5, face.y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
}

int main(int argc, const char * argv[])
{
    VideoCaptureManager capture;
    AlgorithmHolder detector;
    FaceDetector face;

    cv::namedWindow("Capture", CV_WINDOW_OPENGL|CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);

    //! assigns callback for mouse events
    cv::setMouseCallback("Capture", mouseCallback);

    cv::Mat_< unsigned char > frame;



    while(1) {

        capture.read(frame);

        if(!frame.empty()){

//            detector.detect(frame);

            face.detect(frame);

            if(cv::waitKey( 30 ) > 27){
                std::cerr << detector.select() << std::endl;
            }
            cv::imshow( "Capture", frame);

        }
    }
}

