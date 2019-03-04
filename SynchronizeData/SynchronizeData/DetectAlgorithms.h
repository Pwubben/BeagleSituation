#include <memory>
using namespace std;
using namespace cv;

void SaliencyDetect(cv::VideoCapture capture, vector<vector<Rect>> &boundRectVec, double &avg_time, double max_dimension, double sample_step, double stdThres);

void GMMDetect(cv::VideoCapture capture, vector<Rect> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio);

void GroundTruth(cv::VideoCapture capture, vector<Rect> &boundRectVec);

class MyBase {
public:
	MyBase() = default; };
class Sphere : public MyBase {
public:
	Sphere() { } };
class Plane : public MyBase {
public:
	Plane() {} };

class Ext {
public:
	Ext() {};
	~Ext() {};
	void setV() {
		v.push_back(std::unique_ptr<MyBase>(new Sphere()));
		v.push_back(std::unique_ptr<MyBase>(new Plane()));
	}
private:
	std::vector<std::unique_ptr<MyBase> > v;
};

 