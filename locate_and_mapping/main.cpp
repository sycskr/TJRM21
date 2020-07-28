#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);
const float confidenceThreshold = 0.5;

void face_detect_dnn();


int main(int argc, char** argv)
{
    face_detect_dnn();
    waitKey(0);
    return 0;
}

/* detectAndDraw
 * params @ frame(扫描图像) net(网络结构) faces(输出的脸的二维坐标)
 * func @ 检测图像中的人脸（在输入图像上做好标记）并返回二维坐标
 * */
void detectAndDraw( Mat& frame, dnn::Net net,vector<Rect>& faces){
    faces.clear();
    int64 start = getTickCount();

    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);

    /*输入数据调整*/
    Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                  Size(inWidth, inHeight), meanVal, false, false);
    net.setInput(inputBlob, "data");

    /*人脸检测*/
    Mat detection = net.forward("detection_out");

    vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;

    /*投影到人脸检测到的图像上*/
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    ostringstream ss;
    int count =0;
    for (int i = 0; i < detectionMat.rows; i++){
        /*置信度 0～1之间*/
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold){
            /*找到角点*/
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            /*人脸矩形*/
            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

            /*载入容器*/
            faces.push_back(object);

            /*框脸*/
            rectangle(frame, object, Scalar(0, 255, 0));

            cout<<"face"<<++count<<endl;
            ss << confidence;
            String conf(ss.str());
            String label = "Face: " + conf;

            int baseLine = 0;
            /*打印出人脸标号及置信度*/
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                  Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(255, 255, 255), FILLED);

            putText(frame, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }
    }

    float fps = getTickFrequency() / (getTickCount() - start);
    ss.str("");
    ss << "FPS: " << fps << " ; inference time: " << time << " ms";
    putText(frame, ss.str(), Point(20, 20), 0, 0.75, Scalar(0, 0, 255), 2, 8);

}
/*
 * */
string deleteAllMarks(string& str, const string& mark){
    size_t len = mark.length();
    while (1){
        size_t pos = str.find(mark);
        if (pos == string::npos){
            return str;
        }
        str.erase(pos, len);
    }
}

/*inputCalibration
 * params @ dir(输入的相机参数文本路径) cameraMatrix(相机矩阵) distCoeffs(畸变参数)
 * func @ 从指定路径读入相机外参
 * return @
 * */
int inputCalibration(const string & dir,Mat &cameraMatrix,Mat &distCoeffs ){
    std::cout << "准备读取相机标定参数，按任意键继续" << endl;
    waitKey(0);
    cameraMatrix = Mat(3, 3, CV_64FC1);
    distCoeffs = Mat(1, 5, CV_64FC1);
    vector<double> cameraMatrixVector;
    vector<double> distCoeffsVector;

    string line;
    ifstream in(dir);  //读入文件
    if(!in){
        std::cout << "未找到相应相机标定文件，按任意键继续" << endl;
        waitKey(0);
        return -1;
    }
    int i,j;
    for(i=0;getline(in, line);i++){
        deleteAllMarks(line,"[");
        char *str = (char *)line.c_str();//string --> char
        const char *split = ",";
        char *p = strtok (str,split);//逗号分隔依次取出
        double a;
        for(;p != NULL;){
            sscanf(p, "%lf", &a);
            cout<<a<<endl;
            p = strtok(NULL,split);
            if(i<3){
                cameraMatrixVector.push_back(a);
            }
            else{
                distCoeffsVector.push_back(a);
            }
        }
    }

    for(i=0;i<3;i++){
        for(j=0;j<3;j++)
            cameraMatrix.at<double>(i,j)=cameraMatrixVector[i*3+j];
    }
    for(i=0;i<5;i++){
        distCoeffs.at<double>(0,i)=distCoeffsVector[i];
    }//calibrateLists
    cout << "相机内参数矩阵：" << endl;
    cout<<cameraMatrix<<endl;
    cout << "畸变系数："<<endl;
    cout<<distCoeffs<<endl;

    std::cout << "相机标定参数已传入，按任意键继续" << endl;
    waitKey(0);
    return 1;
}

/*PNPsolver
 * params @ dir(输入的相机参数文本路径) cameraMatrix(相机矩阵) distCoeffs(畸变参数)
 * func @ 从指定路径读入相机外参
 * return @
 * */
bool PNPsolver(const std::vector<cv::Point2f>& img,
               const Mat& cameraMatrix,const Mat& distCoeffs,
               double &distance, std::vector<double>&angels, std::vector<double>&euroangels)
{
    /*object size*/
    const double halfwidth =  145 / 2.0;
    const double halfheight = 210 / 2.0;
    std::vector<Point3f> obj
            {
                    Point3f(-halfwidth,  halfheight, 0),   //tl
                    Point3f( halfwidth,  halfheight, 0),   //tr
                    Point3f( halfwidth, -halfheight, 0),   //br
                    Point3f(-halfwidth, -halfheight, 0)    //bl
            };

    Mat rVec = Mat::zeros(3, 1, CV_64FC1);
    Mat tVec = Mat::zeros(3, 1, CV_64FC1);

    /*具体pnp相机姿位结算*/
    if (!solvePnP(obj, img, cameraMatrix, distCoeffs, rVec, tVec, false, SOLVEPNP_P3P))
        return false;

    Mat_<double> rotMat(3, 3);
    /*罗德里格斯变换*/
    Rodrigues(rVec, rotMat);

    /*获得欧拉角*/
    euroangels.push_back(atan2(rotMat[2][1], rotMat[2][2]) * 57.2958);
    euroangels.push_back(atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 57.2958);
    euroangels.push_back(atan2(rotMat[1][0], rotMat[0][0]) * 57.2958);


    double x = tVec.at<double>(0, 0);
    double y = tVec.at<double>(1, 0);
    double z = tVec.at<double>(2, 0);

    /*获得俯仰角*/
    angels.push_back(atan2(x, z));//angels[0]= atan2(x, z);
    angels.push_back(atan2(y, sqrt(x * x + z * z)));//angels[1]= atan2(y, sqrt(x * x + z * z));

    /*获得距离*/
    distance = sqrt(x * x + y * y + z * z);

    return true;
}

/*cal_angle_distance
 * params @ faces(脸部二维矩形框)
 *          cameraMatrix(相机矩阵) distCoeffs(畸变参数)
 *          frame(进行角度距离标记了的图像) faces_position(获得脸的三维坐标)
 * func @ 通过pnp算法，由图像获得矩形框的三维坐标
 * return @
 * */
bool cal_angle_distance(const vector<Rect>& faces,
                        const Mat& cameraMatrix,const Mat& distCoeffs,
                        Mat & frame,vector<Point3d>& faces_position){

    faces_position.clear();
    /*按照单个脸矩形框进行遍历*/
    for (int i=0;i<faces.size();i++) {
        double distance;
        vector<double> angle;
        vector<double> eruangle;

        /*获取当前脸*/
        Rect cur_face=faces[i];

        /*获得top-left bottom-right 和 center 的点坐标*/
        Point2f tl = cur_face.tl();
        Point2f br = cur_face.br();
        Point2f center = Point2f((tl.x+br.x)/2,(tl.y+br.y)/2);

        /*输入的顺序为 ： tl tr rb lb*/
        vector<Point2f> vertice;
        vertice.push_back(cur_face.tl());
        vertice.push_back(cur_face.br());
        vertice.emplace_back(tl.x+cur_face.width,tl.y);
        vertice.emplace_back(br.x-cur_face.width,br.y);

        /*姿位结算 获得距离、角度、欧拉角等 其实可以直接获得三维坐标，不需要进行额外的运算*/
        if(!PNPsolver(vertice,cameraMatrix,distCoeffs,distance,angle,eruangle)){
            return false;
        }
        /*水平偏角*/
        double HA=angle[0];/*单位是rad，不是°*/
        /*竖直偏角*/
        double VA=angle[1];
        double dist=distance/1000;
        cout<<distance<<"   "<<angle[0]<<"   "<<angle[1]<<endl;

        /*打印相应的竖直*/
        std::string msg = cv::format("D:%.2f", dist);//打印内容
        cv::Point textOrigin1(center.x - 20,center.y+60);//打印位置
        cv::putText(frame, msg, textOrigin1, 1, 1, cv::Scalar(0, 255, 0));//实现打印

        msg = cv::format("HA:%.2f ",HA/3.14*180);
        Point textOrigin2(center.x - 20,center.y+60+20);
        cv::putText(frame, msg, textOrigin2, 1, 1, cv::Scalar(0, 255, 0));

        msg = cv::format("VA:%.2f ", -VA/3.14*180);
        cv::Point textOrigin3(center.x - 20,center.y+60+40);
        cv::putText(frame, msg, textOrigin3, 1, 1, cv::Scalar(0, 255, 0));
        //cout<<"sin"<<sin(3.14)<<endl;
        Point3d face_position(dist*cos(VA),dist*cos(VA)*sin(HA),-dist*sin(VA));

        msg = cv::format("(x,y,z)=(%.2f,%.2f,%.2f) ",face_position.x,face_position.y,face_position.z);
        cv::Point textOrigin4(center.x - 20,center.y+60+40+20);
        cv::putText(frame, msg, textOrigin4, 1, 1, cv::Scalar(0, 255, 0));

        /*返回三维坐标*/
        faces_position.push_back(face_position);
    }

}

/*trans_to_flat
 * params @ img(读入原来的场地图) faces_position(输入的脸三维坐标)
 * func @ 通过三维坐标实现对二维场地的投影
 * return @
 * */
bool trans_to_flat(const Mat & img ,const vector<Point3d>& faces_position){
    Mat court = img.clone();

    /*场地大小*/
    double court_length = 1.4/2;
    double court_width  = 1.5/2;

    /*相机在场地的位置*/
    Point offset(court.cols/2,50);
    circle(court,offset,7,Scalar(0,255,0),3);

    /*按脸进行遍历*/
    for(int i = 0; i<faces_position.size() ; i++){
        /*获取当前脸
         * x 水平的前后 y 表示水平的左右 z表示竖直方向
         */
        Point3d cur_player_3Dpst=faces_position[i];
        /*进行坐标的转换，注意场地图和相机获得的脸的 x,y 的区别*/
        double scale_x=cur_player_3Dpst.y/(court_width/2)*(court.cols/2);
        double scale_y=cur_player_3Dpst.x/(court_length)*(court.rows/2);

        /*投影反馈二维位置*/
        Point current_play(offset.x+scale_x,offset.y+scale_y);
        circle(court,current_play,12,Scalar(0,255,0),8);

    }
    imshow("2d_court",court);
    return true;
}

/*face_detect_dnn
 * func @ 相当于检测主函数 进行准备工作与摄像头读取操作，并实现主体功能
 * return @
 * */
void face_detect_dnn() {
    /*配置文件*/
    String modelDesc = "../dnn_face/deploy.prototxt";//opencv_face_detector.pbtxt
    String modelBinary = "../dnn_face/res10_300x300_ssd_iter_140000_fp16.caffemodel";//opencv_face_detector_uint8.pb

    /*初始化网络*/
    dnn::Net net = readNetFromCaffe(modelDesc, modelBinary);//readNetFromTensorflow(modelBinary, modelDesc);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    if (net.empty()){
        printf("could not load net...\n");
        return;
    }

    /*打开摄像头*/
    VideoCapture capture("/dev/video0");
    if (!capture.isOpened()) {
        printf("could not load camera...\n");
        return;
    }

    /*读入相机矩阵*/
    Mat_<double > cameraMatrix,distCoeffs;

    if(!inputCalibration("../dnn_face/calibrateLists.txt",cameraMatrix,distCoeffs)){
        cout<<"读取标定参数失败"<<endl;
    }

    Mat frame;
    Mat court=imread("../court.jpeg");

    /*存放由检测返回的二维坐标*/
    vector<Rect> faces;
    /*存放由pnp解算的三维坐标*/
    vector<Point3d> faces_position;

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;
            Mat frame1 = frame.clone();

            /*二维检测与回显*/
            detectAndDraw( frame, net,faces);

            /*解算获得角度与距离*/
            if(cal_angle_distance(faces,cameraMatrix,distCoeffs,frame,faces_position)){
                return ;
            }

            /*三维到二维平面图的投影*/
            trans_to_flat(court ,faces_position);

            imshow("dnn_face_detection", frame);

            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
}