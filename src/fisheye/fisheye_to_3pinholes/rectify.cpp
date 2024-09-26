// Code in this file was authored by Dr. Zhenyou Dai.

#include <iostream>
#include <map>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

enum CalibMeans
{
	CALIB_FROM_DEF,
	CALIB_FROM_FACT,
	CALIB_FROM_USER,
	CALBI_FROM_SELF
};

enum CalibModel
{
	CALIB_PH8 = 0,
	CALIB_KB4 = 1,
	CALIB_EUCM = 2,
	CALIB_DSCM = 3
};

enum RectiModel
{
	RECTI_PH,
	RECTI_EPED
};

struct BinoParam
{
	CalibMeans means;
	int srcRows;    // original image rows, fisheye 1120, pinhole 1392
	int srcCols;    // original image cols, fisheye 1120, pinhole 976
	int dstRows;    // rectify image rows, determine recording to fov and fx/fy
	int dstCols;    // rectify image cols, determine recording to fov and fx/fy
	double srcParam1[9];    // left camera intrinsic params: fx, fy, cx, cy, k1, k2, p1, p2, k3
	double srcParam2[9];    // right camera intrinsic params: fx, fy, cx, cy, k1, k2, p1, p2, k3
	double dstParam1[4];    // left rectify camera's intrinsic params: fx, fy, cx, cy
	double dstParam2[4];    // right rectify camera's intrinsic params: fx, fy, cx, cy
	double trans[3];    // extrinsic params transtion (T_R_L)
	double rvec[3];
	double quat[4];//xyzw
	double rmat[9];     // extrinsic params rotation (R_R_L), the rest attributes can be ignored
	double euler[3];
	double rvec1[3];
	double quat1[4];//xyzw
	double rmat1[9];
	double euler1[3];
	double rvec2[3];
	double quat2[4];//xyzw
	double rmat2[9];
	double euler2[3];
	CalibModel calibModel;
	RectiModel rectiModel;
	struct RectiCamera { CalibModel _model; RectiModel model; int cols, rows; double params[4]; double axis; };
	union
	{
		RectiCamera multiRectis[3];
		int8_t data[127 * 8] = { 0 };
	} extension;
};

Eigen::Vector3d rotation2euler(const Eigen::Matrix3d &R) {
    double X, Y, Z;
    double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6;
    if(!singular) {
        X = atan2(R(2, 1), R(2, 2));
        Y = atan2(-R(2, 0), sy);
        Z = atan2(R(1, 0), R(0, 0));
    } else {
        X = atan2(-R(1, 2), R(1, 1));
        Y = atan2(-R(2, 0), sy);
        Z = 0;
    }
    Eigen::Vector3d result = {X, Y, Z};
    return result * 180 / M_PI;
}

template <typename M33>
static inline void cvtVec9ToMatx33d(double vec[9], M33& matx) {
    // clang-format off
    matx(0, 0) = vec[0]; matx(0, 1) = vec[1]; matx(0, 2) = vec[2];
    matx(1, 0) = vec[3]; matx(1, 1) = vec[4]; matx(1, 2) = vec[5];
    matx(2, 0) = vec[6]; matx(2, 1) = vec[7]; matx(2, 2) = vec[8];
    // clang-format on
}

template <typename M33>
static inline void cvtVec9FromMatx33d(double vec[9], const M33& matx) {
    // clang-format off
    vec[0] = matx(0, 0); vec[1] = matx(0, 1); vec[2] = matx(0, 2);
    vec[3] = matx(1, 0); vec[4] = matx(1, 1); vec[5] = matx(1, 2);
    vec[6] = matx(2, 0); vec[7] = matx(2, 1); vec[8] = matx(2, 2);
    // clang-format on
}

template <int Len, typename VecX>
static inline void cvtVecXFromVecX(double* vec, const VecX& vec2) {
    for (auto i = 0; i < Len; i++) {
        vec[i] = vec2(i);
    }
}

void GenerateIdealRot(const Eigen::Matrix3d& res_r, const Eigen::Vector3d& res_t,
                      cv::Mat_<double>& rmat1, cv::Mat_<double>& rmat2) {
    double rot_vec[9];
    cvtVec9FromMatx33d(rot_vec, res_r);

    double trans_vec[3];
    cvtVecXFromVecX<3>(trans_vec, res_t);
    cv::Mat_<double> irmat = cv::Mat_<double>(3, 3, rot_vec).t();
    cv::Mat_<double> itrans = -irmat * cv::Mat_<double>(3, 1, trans_vec);
    rmat1.create(3, 3);
    rmat1.eye(3, 3);  // no axis rotation
    cv::Mat e1, e2, e3;
    e1 = itrans.t() / cv::norm(itrans);
    e2 = cv::Mat_<double>({1, 3}, {-itrans(1), itrans(0), 0});
    e2 = e2 / cv::norm(e2);
    e3 = e1.cross(e2);
    e3 = e3 / cv::norm(e3);
    e1.copyTo(rmat1.row(0));
    e2.copyTo(rmat1.row(1));
    e3.copyTo(rmat1.row(2));
    rmat2 = rmat1 * irmat;
}

void RemapCamera(int calib_model, int recti_model, double* calib_param, double* recti_param,
                 double* rmat, cv::Mat_<float>& mapx, cv::Mat_<float>& mapy) {
    cv::Matx33d irot = cv::Matx33d(rmat).inv();
    for (int v = 0; v < mapx.rows; ++v) {
        for (int u = 0; u < mapx.cols; ++u) {
            // recti_model
            cv::Vec3d P3D(0, 0, 1);
            double xn = (u - recti_param[2]) / recti_param[0];
            double yn = (v - recti_param[3]) / recti_param[1];
            std::cout<<(int)recti_model <<std::endl;
            if (recti_model == RECTI_PH)
                P3D = cv::Vec3d(xn, yn, 1);
            else {
                P3D[0] = sin(xn);
                P3D[1] = cos(xn) * sin(yn);
                P3D[2] = cos(xn) * cos(yn);
            }
            P3D = irot * P3D;
            const double& x = P3D[0];
            const double& y = P3D[1];
            const double& z = P3D[2];

            // calib_model
            const double& fx = calib_param[0];
            const double& fy = calib_param[1];
            const double& cx = calib_param[2];
            const double& cy = calib_param[3];
            double mx = 0, my = 0;
            if (calib_model == CALIB_PH8) {
                const double& k1 = calib_param[4];
                const double& k2 = calib_param[5];
                const double& p1 = calib_param[6];
                const double& p2 = calib_param[7];
                const double& k3 = calib_param[8];
                double iz = 1. / z;
                double xn = x * iz;
                double yn = y * iz;
                double xn2 = xn * xn;
                double yn2 = yn * yn;
                double rn2 = xn2 + yn2;
                double rn4 = rn2 * rn2;
                double rn6 = rn2 * rn4;
                double xnyn = 2. * xn * yn;
                double r2xn2 = rn2 + 2. * xn2;
                double r2yn2 = rn2 + 2. * yn2;
                double rd = 1. + k1 * rn2 + k2 * rn4 + k3 * rn6;
                mx = xn * rd + p1 * xnyn + p2 * r2xn2;
                my = yn * rd + p2 * xnyn + p1 * r2yn2;
            } else if (calib_model == CALIB_KB4) {
                const double& k1 = calib_param[4];
                const double& k2 = calib_param[5];
                const double& k3 = calib_param[6];
                const double& k4 = calib_param[7];
                const double r2 = x * x + y * y;
                const double r = sqrt(r2);
                const double theta = atan2(r, z);
                const double theta2 = theta * theta;
                double r_theta = k4 * theta2;
                r_theta += k3;
                r_theta *= theta2;
                r_theta += k2;
                r_theta *= theta2;
                r_theta += k1;
                r_theta *= theta2;
                r_theta += 1;
                r_theta *= theta;
                mx = x * r_theta / r;
                my = y * r_theta / r;
            } else if (calib_model == CALIB_EUCM) {
                const double& alpha = calib_param[4];
                const double& beta = calib_param[5];
                const double r2 = x * x + y * y;
                const double rho2 = beta * r2 + z * z;
                const double rho = sqrt(rho2);
                const double norm = alpha * rho + (1. - alpha) * z;
                mx = x / norm;
                my = y / norm;
            }
            mapx(v, u) = fx * mx + cx;
            mapy(v, u) = fy * my + cy;
        }
    }
}

void GenerateDataFromBinoParam(BinoParam& bp, const BinoParam& fact_bp,
                               const Eigen::Matrix3d& res_r, const Eigen::Vector3d& res_t,
                               cv::Mat& tablexy1, cv::Mat& tablexy2) {
    // 1. generate param
    // keep srcsize, rectisize, srcparam, dstparam the same
    bp = fact_bp;
    // 1.1 update R, T
    cvtVecXFromVecX<3>(bp.trans, res_t * 100);  // cm to m
    cvtVec9FromMatx33d(bp.rmat, res_r);
    auto Q = Eigen::Quaterniond(res_r);
    cvtVecXFromVecX<4>(bp.quat, Q.coeffs());
    cvtVecXFromVecX<3>(
        bp.rvec, -Eigen::AngleAxisd(res_r).axis() * (M_PI * 2 - Eigen::AngleAxisd(res_r).angle()));
    cvtVecXFromVecX<3>(bp.euler, rotation2euler(res_r));

    // 1.2 Get r1 r2
    cv::Mat_<double> rmat1, rmat2;
    GenerateIdealRot(res_r, res_t, rmat1, rmat2);

    // 1.3 transfer to Binoparam
    cvtVec9FromMatx33d(bp.rmat1, rmat1);
    Eigen::Matrix3d rmat1_eg, rmat2_eg;
    cvtVec9ToMatx33d(bp.rmat1, rmat1_eg);
    auto Q1 = Eigen::Quaterniond(rmat1_eg);
    cvtVecXFromVecX<4>(bp.quat1, Q1.coeffs());
    cvtVecXFromVecX<3>(bp.rvec1, -Eigen::AngleAxisd(rmat1_eg).axis() *
                                     (M_PI * 2 - Eigen::AngleAxisd(rmat1_eg).angle()));
    cvtVecXFromVecX<3>(bp.euler1, rotation2euler(rmat1_eg));

    cvtVec9FromMatx33d(bp.rmat2, rmat2);
    cvtVec9ToMatx33d(bp.rmat2, rmat2_eg);
    auto Q2 = Eigen::Quaterniond(rmat2_eg);
    cvtVecXFromVecX<4>(bp.quat2, Q2.coeffs());
    cvtVecXFromVecX<3>(bp.rvec2, -Eigen::AngleAxisd(rmat2_eg).axis() *
                                     (M_PI * 2 - Eigen::AngleAxisd(rmat2_eg).angle()));
    cvtVecXFromVecX<3>(bp.euler2, rotation2euler(rmat2_eg));

    // 2. generate table (one to one)
    if (fact_bp.extension.multiRectis[0].rows == 0) {
        // 2.1 one fisheye to one equi-dist
        std::cout << "One fisheye to one cam!" << std::endl;
        // auLog_info(AU_LOG_OUTPUT_BOTH, "One fisheye to one cam!");
        cv::Mat_<float> mapx1, mapy1, mapx2, mapy2;
        mapx1.create(bp.dstRows, bp.dstCols);
        mapy1.create(bp.dstRows, bp.dstCols);
        mapx2.create(bp.dstRows, bp.dstCols);
        mapy2.create(bp.dstRows, bp.dstCols);
        const int calib_model = fact_bp.calibModel;
        const int recti_model = fact_bp.rectiModel;

        // keep same recti cam intrinsic and img size as factory setup
        RemapCamera(calib_model, recti_model, bp.srcParam1, bp.dstParam1, rmat1.ptr<double>(),
                    mapx1, mapy1);
        RemapCamera(calib_model, recti_model, bp.srcParam2, bp.dstParam2, rmat2.ptr<double>(),
                    mapx2, mapy2);

        // 2.2 merge table
        cv::merge(std::vector<cv::Mat>{mapx1, mapy1}, tablexy1);
        cv::merge(std::vector<cv::Mat>{mapx2, mapy2}, tablexy2);
        // 3. generate table (one to multi)
    } else {
        std::vector<cv::Mat> vmapx1, vmapy1, vmapx2, vmapy2;
        int final_recti_rows = 0, final_recti_cols = 0;
        // 3.1 one fisheye to multi equi-dist
        int recti_cam_num = sizeof(fact_bp.extension.multiRectis) / sizeof(BinoParam::RectiCamera);
        std::cout << "One fisheye to " << recti_cam_num << " cams" << std::endl;
        // auLog_info(AU_LOG_OUTPUT_BOTH, "One fisheye to %d cam!", recti_cam_num);
        for (auto i = 0; i < recti_cam_num; i++) {
            auto multi_params = fact_bp.extension.multiRectis[i];
            const int calib_model = multi_params._model;
            const int recti_model = multi_params.model;

            const double angle_axis = multi_params.axis * M_PI / 180;
            // clang-format off
            cv::Mat_<double> rmat0({3, 3}, {1, 0, 0, 
                                            0, cos(angle_axis), -sin(angle_axis),
                                            0, sin(angle_axis), cos(angle_axis)});  // rotation around x
            // clang-format on
            cv::Mat_<double> rmat1_rot, rmat2_rot;
            rmat1_rot = rmat0 * rmat1;
            rmat2_rot = rmat0 * rmat2;

            // generate table
            cv::Mat_<float> mapx1, mapy1, mapx2, mapy2;
            mapx1.create(multi_params.rows, multi_params.cols);
            mapy1.create(multi_params.rows, multi_params.cols);
            mapx2.create(multi_params.rows, multi_params.cols);
            mapy2.create(multi_params.rows, multi_params.cols);

            // keep same recti cam intrinsic and img size as factory setup
            RemapCamera(calib_model, recti_model, bp.srcParam1, multi_params.params,
                        rmat1_rot.ptr<double>(), mapx1, mapy1);
            RemapCamera(calib_model, recti_model, bp.srcParam2, multi_params.params,
                        rmat2_rot.ptr<double>(), mapx2, mapy2);

            vmapx1.push_back(mapx1);
            vmapy1.push_back(mapy1);
            vmapx2.push_back(mapx2);
            vmapy2.push_back(mapy2);
            // concat Map
            final_recti_rows += multi_params.rows;
            if (final_recti_cols < multi_params.cols) {
                final_recti_cols = multi_params.cols;
            }
        }
        // 3.2 concate tables
        cv::Mat_<float> final_mapx1(final_recti_rows, final_recti_cols, 0.f);
        cv::Mat_<float> final_mapx2(final_recti_rows, final_recti_cols, 0.f);
        cv::Mat_<float> final_mapy1(final_recti_rows, final_recti_cols, 0.f);
        cv::Mat_<float> final_mapy2(final_recti_rows, final_recti_cols, 0.f);
        for (size_t k = 0, irow = 0; k < vmapx1.size(); ++k) {
            vmapx1[k].copyTo(
                final_mapx1.rowRange(irow, irow + vmapx1[k].rows).colRange(0, vmapx1[k].cols));
            vmapx2[k].copyTo(
                final_mapx2.rowRange(irow, irow + vmapx2[k].rows).colRange(0, vmapx2[k].cols));
            vmapy1[k].copyTo(
                final_mapy1.rowRange(irow, irow + vmapy1[k].rows).colRange(0, vmapy1[k].cols));
            vmapy2[k].copyTo(
                final_mapy2.rowRange(irow, irow + vmapy2[k].rows).colRange(0, vmapy2[k].cols));
            irow += vmapx1[k].rows;
        }
        // 3.3 merge table
        cv::merge(std::vector<cv::Mat>{final_mapx1, final_mapy1}, tablexy1);
        cv::merge(std::vector<cv::Mat>{final_mapx2, final_mapy2}, tablexy2);
    }
}

// Below are yaml related functions

template<typename T> static auto Model2Name(T src, bool calib)
{
    static const std::map<int, std::string> cmnames
    {
        { CALIB_PH8, "pinhole-radtan8" },
        { CALIB_KB4, "kb4" },
        { CALIB_EUCM, "eucm" }
    };
    static const std::map<int, std::string> rmnames
    {
        { RECTI_PH, "PH" },
        { RECTI_EPED, "EPED" }
    };
    if (calib)
    {
        if constexpr (std::is_null_pointer<T>::value) return cmnames;
        if constexpr (std::is_same<T, int>::value) return cmnames.at(src);
        if constexpr (std::is_same<T, std::string>::value)
            return std::find_if(cmnames.begin(), cmnames.end(), [&src](const std::pair<int, std::string>& v) { return v.second == src; })->first;
    }
    else
    {
        if constexpr (std::is_null_pointer<T>::value) return rmnames;
        if constexpr (std::is_same<T, int>::value) return rmnames.at(src);
        if constexpr (std::is_same<T, std::string>::value)
            return std::find_if(rmnames.begin(), rmnames.end(), [&src](const std::pair<int, std::string>& v) { return v.second == src; })->first;
    }
}

void Readyaml(std::string path, BinoParam& params) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    params.means = CALIB_FROM_FACT;
    std::string cmname; fs["calibModel"] >> cmname;
    std::string rmname; fs["rectiModel"] >> rmname;
    params.calibModel = CalibModel(Model2Name(cmname, true));
    params.rectiModel = RectiModel(Model2Name(rmname, false));
    fs["calibCols"] >> params.srcCols;
    fs["calibRows"] >> params.srcRows;
    fs["rectiCols"] >> params.dstCols;
    fs["rectiRows"] >> params.dstRows;

    auto SingleRowMat2Array = [](cv::Mat_<double> m, double* a) {
        for (int i = 0; i < m.cols; ++i) {
            a[i] = m.at<double>(0, i);
        } 
    };

    cv::Mat_<double> cvcalibParam1, cvcalibParam2;
    fs["calibParam1"] >> cvcalibParam1;
    fs["calibParam2"] >> cvcalibParam2;

    SingleRowMat2Array(cvcalibParam1, params.srcParam1);
    SingleRowMat2Array(cvcalibParam2, params.srcParam2);

    cv::Mat_<double> cvrectiParam1, cvrectiParam2;
    fs["rectiParam1"] >> cvrectiParam1;
    fs["rectiParam2"] >> cvrectiParam2;

    SingleRowMat2Array(cvrectiParam1, params.dstParam1);
    SingleRowMat2Array(cvrectiParam2, params.dstParam2);

    cv::Mat_<double> cvtrans, cvrvec, cvquat, cvrmat, cveuler;
    fs["trans"] >> cvtrans;
    fs["rvec"] >> cvrvec;
    fs["quat"] >> cvquat;
    fs["euler"] >> cveuler;

    SingleRowMat2Array(cvtrans, params.trans);
    SingleRowMat2Array(cvrvec, params.rvec);
    SingleRowMat2Array(cvquat, params.quat);
    SingleRowMat2Array(cveuler, params.euler);

    fs["rmat"] >> cvrmat;
    cvtVec9FromMatx33d(params.rmat, cvrmat);

    std::vector<std::vector<double>> multiRectis;
    auto nd = fs["multiRectis"];
    if (nd.fs) nd >> multiRectis;

    int num = multiRectis.size();
    for (int k = 0; k < num; ++k)
    {
        params.extension.multiRectis[k]._model = CalibModel(multiRectis[k][0]);
        params.extension.multiRectis[k].model = RectiModel(multiRectis[k][1]);
        params.extension.multiRectis[k].cols = int(multiRectis[k][2]);
        params.extension.multiRectis[k].rows = int(multiRectis[k][3]);
        params.extension.multiRectis[k].params[0] = multiRectis[k][4];
        params.extension.multiRectis[k].params[1] = multiRectis[k][5];
        params.extension.multiRectis[k].params[2] = multiRectis[k][6];
        params.extension.multiRectis[k].params[3] = multiRectis[k][7];
        params.extension.multiRectis[k].axis = multiRectis[k][8];
    }
}

template <int Len, typename VecX>
static inline void cvtVecXToVecX(VecX& vec2, double* vec) {
    vec2.create(1, Len);
    for (auto i = 0; i < Len; i++) {
        vec2(i) = vec[i];
    }
}

void SaveYaml(std::string path, BinoParam& p, const cv::Mat& x1, const cv::Mat& y1,
              const cv::Mat& x2, const cv::Mat& y2) {
    cv::Mat_<double> cvcalibParam1, cvcalibParam2;
    cv::Mat_<double> cvrectiParam1, cvrectiParam2;
    cv::Mat_<double> cvtrans, cvrvec, cvquat, cvrmat, cveuler;
    cv::Mat_<double> cvrvec1, cvquat1, cvrmat1, cveuler1;
    cv::Mat_<double> cvrvec2, cvquat2, cvrmat2, cveuler2;
    cv::Vec4i range1(0), range2(0);
    cv::Mat_<float> mapx1, mapy1;
    cv::Mat_<float> mapx2, mapy2;

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "calibModel" << Model2Name(static_cast<int>(p.calibModel), true);
    fs << "rectiModel" << Model2Name(static_cast<int>(p.rectiModel), false);
    fs << "calibCols" << p.srcCols;
    fs << "calibRows" << p.srcRows;
    fs << "rectiCols" << p.dstCols;
    fs << "rectiRows" << p.dstRows;
    
    cvtVecXToVecX<6>(cvcalibParam1, p.srcParam1);
    cvtVecXToVecX<6>(cvcalibParam2, p.srcParam2);
    cvtVecXToVecX<4>(cvrectiParam1, p.dstParam1);
    cvtVecXToVecX<4>(cvrectiParam2, p.dstParam2);
    fs << "calibParam1" << cvcalibParam1;
    fs << "calibParam2" << cvcalibParam2;
    fs << "rectiParam1" << cvrectiParam1;
    fs << "rectiParam2" << cvrectiParam2;

    cvtVecXToVecX<3>(cvtrans, p.trans);
    cvtVecXToVecX<3>(cvrvec, p.rvec);
    cvtVecXToVecX<4>(cvquat, p.quat);
    cvrmat.create(3,3);
    cvtVec9ToMatx33d(p.rmat, cvrmat);
    cvtVecXToVecX<3>(cveuler, p.euler);

    fs << "trans" << cvtrans;
    fs << "rvec" << cvrvec;
    fs << "quat" << cvquat;
    fs << "rmat" << cvrmat;
    fs << "euler" << cveuler;

    cvtVecXToVecX<3>(cvrvec1, p.rvec1);
    cvtVecXToVecX<4>(cvquat1, p.quat1);
    cvrmat1.create(3,3);
    cvtVec9ToMatx33d(p.rmat1, cvrmat1);
    cvtVecXToVecX<3>(cveuler1, p.euler1);

    fs << "rvec1" << cvrvec1;
    fs << "quat1" << cvquat1;
    fs << "rmat1" << cvrmat1;
    fs << "euler1" << cveuler1;

    cvtVecXToVecX<3>(cvrvec2, p.rvec2);
    cvtVecXToVecX<4>(cvquat2, p.quat2);
    cvrmat2.create(3,3);
    cvtVec9ToMatx33d(p.rmat2, cvrmat2);
    cvtVecXToVecX<3>(cveuler2, p.euler2);

    fs << "rvec2" << cvrvec2;
    fs << "quat2" << cvquat2;
    fs << "rmat2" << cvrmat2;
    fs << "euler2" << cveuler2;

    // wrong value, maybe 0, doesnt matter
    fs << "range1" << range1;
    fs << "range2" << range2;

    std::vector<std::vector<double>> multiRectis;
    int num = sizeof(p.extension.multiRectis) / sizeof(BinoParam::RectiCamera);
    for (int k = 0; k < num; ++k)
    {
        if (p.extension.multiRectis[k].rows > 0)
        {
            std::vector<double> one;
            one.push_back(p.extension.multiRectis[k]._model);
            one.push_back(p.extension.multiRectis[k].model);
            one.push_back(p.extension.multiRectis[k].cols);
            one.push_back(p.extension.multiRectis[k].rows);
            one.push_back(p.extension.multiRectis[k].params[0]);
            one.push_back(p.extension.multiRectis[k].params[1]);
            one.push_back(p.extension.multiRectis[k].params[2]);
            one.push_back(p.extension.multiRectis[k].params[3]);
            one.push_back(p.extension.multiRectis[k].axis);
            multiRectis.push_back(one);
        }
    }
    if (multiRectis.size() > 0)
    {
        fs << "multiRectis" << multiRectis;
    }

    fs << "mapx1" << x1;
    fs << "mapx2" << x2;
    fs << "mapy1" << y1;
    fs << "mapy2" << y2;

    
}

int main(int argc, char* argv[]) {
     if (argc != 3) {
        std::cout << "usage: ./rectify_tool [input .yml path] [output .yml path"
                  << std::endl;
        return 1;
    }
    BinoParam p;
    Readyaml(argv[1], p);

    // extract trans and rotation matrix
    Eigen::Vector3d trans = Eigen::Vector3d(p.trans) * 0.01;
    Eigen::Matrix3d rot;
    cvtVec9ToMatx33d(p.rmat, rot);

    BinoParam new_p;
    cv::Mat tablexy1, tablexy2;
    GenerateDataFromBinoParam(new_p, p, rot, trans, tablexy1, tablexy2);

    // split tables
    std::vector<cv::Mat> splited_1, splited_2;
    cv::split(tablexy1, splited_1);
    cv::split(tablexy2, splited_2);

    cv::Mat x1, y1, x2, y2;
    x1 = splited_1[0];
    y1 = splited_1[1];
    x2 = splited_2[0];
    y2 = splited_2[1];

    // save
    SaveYaml(argv[2], new_p, x1, y1, x2, y2);
}