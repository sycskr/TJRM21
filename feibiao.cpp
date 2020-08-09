/*代码版本测试*/
#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>


namespace cv
{
    bool PNPsolver(std::vector<cv::Point2f> img, double &distance, std::vector<double>&angels, std::vector<double>&euroangels)
    {
        using namespace cv;

        //飞镖引导灯尺寸object size 单位mm
        const double halfwidth =  5.5 / 2.0;
        const double halfheight = 5.5 / 2.0;
        std::vector<Point3f> obj
                {
                        Point3f(-halfwidth,  halfheight, 0),   //tl左上
                        Point3f( halfwidth,  halfheight, 0),   //tr右上
                        Point3f( halfwidth, -halfheight, 0),   //br右下
                        Point3f(-halfwidth, -halfheight, 0)    //bl左下
                };

        //相机标定参数：
        Mat cameraMatrix = Mat::zeros(3, 3, CV_64FC1);
        cameraMatrix.at<double>(0, 0) = 1386.659138037379;
        cameraMatrix.at<double>(0, 2) = 322.5162041924799;
        cameraMatrix.at<double>(1, 1) = 1389.403381661966;
        cameraMatrix.at<double>(1, 2) = 248.2318998794119;
        cameraMatrix.at<double>(2, 2) = 1;

        Mat distCoeffs = Mat(1, 5, CV_64FC1);
        distCoeffs.at<double>(0, 0) = -0.8705122065775176;
        distCoeffs.at<double>(0, 1) = 37.60755206070051;
        distCoeffs.at<double>(0, 2) = 0.0006281770228791899;
        distCoeffs.at<double>(0, 3) = 0.001964311138785321;
        distCoeffs.at<double>(0, 4) = -826.0164163728507;

        Mat rVec = Mat::zeros(3, 1, CV_64FC1);
        Mat tVec = Mat::zeros(3, 1, CV_64FC1);

        bool p=solvePnP(obj, img, cameraMatrix, distCoeffs, rVec, tVec, false, CV_P3P);
        if (!p)
            return p;

        Mat_<double> rotMat(3, 3);
        Rodrigues(rVec, rotMat);

        euroangels.push_back(atan2(rotMat[2][1], rotMat[2][2]) * 57.2958);
        euroangels.push_back(atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 57.2958);
        euroangels.push_back(atan2(rotMat[1][0], rotMat[0][0]) * 57.2958);

        double x = tVec.at<double>(0, 0);
        double y = tVec.at<double>(1, 0);
        double z = tVec.at<double>(2, 0);

        angels.push_back(atan2(x, z));//angels[0]= atan2(x, z);
        angels.push_back(atan2(y, sqrt(x * x + z * z)));//angels[1]= atan2(y, sqrt(x * x + z * z));
        //cout<<1<<endl;
        //cout<<angels[0]<<endl;

        distance = sqrt(x * x + y * y + z * z);

        if(!p)
            return p;
        return 1;
    }
}

int main(){
//    摄像头模式：
//    cv::VideoCapture cap(0);
//        if (!cap.isOpened())
//    {
//        std::cout << "Cannot open the camera" << std::endl;
//        return -1;
//    }
    //视频模式：
    cv::VideoCapture cap;
    cap.open("/home/syc/桌面/飞镖/out2.avi");


    struct color{
        int iLowH ;
        int iHighH ;
        int iLowH1;
        int iHighH1;
        int iLowS ;
        int iHighS ;
        int iLowV ;
        int iHighV ;
    }project_color[3];

    ////颜色设置 0,1,2红色,绿色,蓝色
    //红色
    project_color[0].iLowH=170;
    project_color[0].iHighH=180;
    project_color[0].iLowH1=0;
    project_color[0].iHighH1=10;
    project_color[0].iLowS=43;
    project_color[0].iHighS=255;
    project_color[0].iLowV=46;
    project_color[0].iHighV=255;
    //绿色
    project_color[1].iLowH=60;
    project_color[1].iHighH=100;
    project_color[1].iLowH1=0;
    project_color[1].iHighH1=0;
    project_color[1].iLowS=90;
    project_color[1].iHighS=255;
    project_color[1].iLowV=90;
    project_color[1].iHighV=255;
    //蓝色
    project_color[2].iLowH=100;
    project_color[2].iHighH=140;
    project_color[2].iLowH1=0;
    project_color[2].iHighH1=0;
    project_color[2].iLowS=90;
    project_color[2].iHighS=255;
    project_color[2].iLowV=90;
    project_color[2].iHighV=255;

    //颜色模式：0,1,2红色,绿色,蓝色
    int NUM=1;

    while(1){

        cv::Mat imgOriginal; 
        bool bSuccess = cap.read(imgOriginal); 
        //cv::resize(img,img,cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        if (!bSuccess) { 
            std::cout << "Cannot read a frame from video stream" << std::endl; 
            break; 
        } 
        cv::Mat imgHSV; 
        std::vector<cv::Mat>hsvSplit;
        cv::cvtColor(imgOriginal,imgHSV,cv::COLOR_BGR2HSV);
        cv::split(imgHSV,hsvSplit);//将图片颜色通道分离
        cv::equalizeHist(hsvSplit[2],hsvSplit[2]);//直方图均衡化，提升对比度
        cv::merge(hsvSplit,imgHSV);//各个通道进行合并
        cv::Mat imgThresholded;
        cv::inRange(imgHSV,cv::Scalar(project_color[NUM].iLowH,project_color[NUM].iLowS,project_color[NUM].iLowV),cv::Scalar(project_color[NUM].iHighH,project_color[NUM].iHighS,project_color[NUM].iHighV),imgThresholded);

         
        
        //形态学处理 去除一些噪点
        cv::Mat element =cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));//定义 kernel 矩阵
        cv::morphologyEx(imgThresholded,imgThresholded,cv::MORPH_OPEN,element);//开运算
        cv::morphologyEx(imgThresholded,imgThresholded,cv::MORPH_CLOSE,element);//闭运算
        //对灰度图进行滤波
        cv::GaussianBlur(imgThresholded,imgThresholded,cv::Size(3,3),0,0);

        //边缘检测
        cv::Mat cannyImage;
        cv::Canny(imgThresholded,cannyImage,128,255,3);//采用 CANNY 算法发现输入图像的边缘而且在输出图像中标识这些边缘

        //轮廓提取
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(cannyImage,contours,hierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE,cv::Point(0,0));
        //mode:cv::RETR_EXTERNAL只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
        // method:cv::CHAIN_APPROX_SIMPLE只检测最外层轮廓，并且保存轮廓上所有点



        //绘制轮廓
        for(int i=0;i<(int)contours.size();i++){
            cv::drawContours(cannyImage,contours,i,cv::Scalar(255),1,8);
        }
        cv::imshow("处理后的图形",cannyImage);
        
        //用矩形圈出轮廓并返回位置坐标
        //设置初始轮廓面积
//        std::vector<cv::Point> pointsBegin=contours[0];
//        cv::RotatedRect boxBegin=cv::minAreaRect(cv::Mat(pointsBegin));
//        cv::Rect boxRectBegin =boxBegin.boundingRect();
//        float AreaBefore=boxRectBegin.area();
        //开始循环
        for(int i=0;i<contours.size();i++){
            //每个轮廓
            std::vector<cv::Point> points=contours[i];
            //对给定的点集，寻找最小面积的包围矩形
            cv::RotatedRect box=cv::minAreaRect(cv::Mat(points));//points
            //筛选：
            cv::Rect boxRect =box.boundingRect();
            if(boxRect.width/boxRect.height>=1.2)continue;
            //如果宽比高长过多，则不画框
            if(boxRect.height/boxRect.width>=1.2)continue;
            //如果高比宽过多，则不画框
            if(box.angle+90>=15&&box.angle+90<=85)continue;
            //如果太斜，则不要
            if(boxRect.area()<200)continue;
            //如果太小，则不要


            std::cout<<"angle:"<<box.angle+90;

            //AreaBefore=boxRect.area();
            // //***************测距***********************

            std::vector<cv::Point2f> Points2fVec;
            cv::Point2f points2f[4];
            box.points(points2f);
            Points2fVec.push_back(points2f[1]);
            Points2fVec.push_back(points2f[2]);
            Points2fVec.push_back(points2f[3]);
            Points2fVec.push_back(points2f[0]);

            std::cout<<Points2fVec<<std::endl;

            std::vector<double>angels;
            std::vector<double>euroangels;
            double distance;


            PNPsolver(Points2fVec,distance,angels,euroangels);
            if(distance==0)continue;//bug：打印文字的时候，遇到有一处distance=0会直接闪退
            std::cout<<"distance:"<<distance<<std::endl;

            // //*****************************************

            
            double num=boxRect.width/boxRect.height*1.000000;
            //std::cout<<num<<std::endl;
            if(num>1)continue;
            
            cv::Point2f vertex[4];
            box.points(vertex);
            //绘制
            cv::line(imgOriginal,vertex[0],vertex[1],cv::Scalar(100,200,211),6,cv::LINE_AA);
            cv::line(imgOriginal,vertex[1],vertex[2],cv::Scalar(100,200,211),6,cv::LINE_AA);
            cv::line(imgOriginal,vertex[2],vertex[3],cv::Scalar(100,200,211),6,cv::LINE_AA);
            cv::line(imgOriginal,vertex[3],vertex[0],cv::Scalar(100,200,211),6,cv::LINE_AA);

            //绘制中心
            cv::Point s1,l,r,u,d;
            s1.x=((vertex[0].x+vertex[2].x)/2.0);
            s1.y=((vertex[0].y+vertex[2].y)/2.0);
            l.x=s1.x-10;
            l.y=s1.y;

            r.x=s1.x+10;
            r.y=s1.y;

            u.x=s1.x;
            u.y=s1.y-10;

            d.x=s1.x;
            d.y=s1.y+10;
            cv::line(imgOriginal,l,r,cv::Scalar(100,200,211),2,cv::LINE_AA);
            cv::line(imgOriginal,u,d,cv::Scalar(100,200,211),2,cv::LINE_AA);

            //测距
            std::string msg = cv::format("D:%.2f", distance / 1000);
            ////打在框下方
//            cv::Point textOrigin1(box.center.x - 20, box.center.y+60 );
//            cv::putText(imgOriginal, msg, textOrigin1, 1, 1, cv::Scalar(0, 255, 0));
//
//            msg = cv::format("HA:%.2f ", angels[0] / 3.14 * 180);
//            cv::Point textOrigin2(box.center.x - 20, box.center.y +60 + 20);
//            cv::putText(imgOriginal, msg, textOrigin2, 1, 1, cv::Scalar(0, 255, 0));
//
//            msg = cv::format("VA:%.2f ", -angels[1] / 3.14 * 180);
//            cv::Point textOrigin3(box.center.x - 20, box.center.y  +60+ 40);
//            cv::putText(imgOriginal, msg, textOrigin3, 1, 1, cv::Scalar(0, 255, 0));

            ////打在左上角
            cv::Point textOrigin1(20, 60 );
            cv::putText(imgOriginal, msg, textOrigin1, 1, 1, cv::Scalar(0, 255, 0));

            msg = cv::format("HA:%.2f ", angels[0] / 3.14 * 180);
            cv::Point textOrigin2(20, 60 + 20);
            cv::putText(imgOriginal, msg, textOrigin2, 1, 1, cv::Scalar(0, 255, 0));

            msg = cv::format("VA:%.2f ", -angels[1] / 3.14 * 180);
            cv::Point textOrigin3( 20, 60+ 40);
            cv::putText(imgOriginal, msg, textOrigin3, 1, 1, cv::Scalar(0, 255, 0));
            
        }
        cv::namedWindow("绘制",cv::WINDOW_NORMAL);

        cv::imshow("绘制",imgOriginal);
        char key =(char)cv::waitKey(30);
        if(key==27) break;

    }

}