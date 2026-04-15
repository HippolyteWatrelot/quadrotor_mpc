#ifndef LINEAR_MPC_H_INCLUDED
#define LINEAR_MPC_H_INCLUDED


#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <assert.h>
#include <numeric>
#include <random>
#include <iterator>
#include <ros/ros.h>
#include <ros/node_handle.h>

#include <casadi/casadi.hpp>

#include "std_msgs/Int16.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/MultiArrayDimension.h"

#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/Twist.h"

#include "quadrotor_mpc/quaternion.h"




class MPC
{



public:
    MPC(bool gt, bool w);
    ~MPC();
    std::vector<int> T;
    std::vector<int> control_dim;
    casadi::SX AK, C;
    casadi::SX Q, R, P;
    void cost_matrices();
    static Eigen::MatrixXd matrix_power(Eigen::MatrixXd M, int n);
    void build_fixed_point(bool test=false, double speed=0.5, int margin=100);
    
    void init_x0_u0(bool init_command);
    Eigen::Vector3d forward_body_twist_w(double& roll, double& pitch, double& yaw, double& rolld, double& pitchd, double& yawd);
    Eigen::Vector3d forward_body_twist_command_w(casadi::DM& state, casadi::DM& command);
    
    Eigen::Vector3d quaternion_to_euler(Eigen::Vector4d& q) const;
    
    /*casadi::SX VecToso3(casadi::SX vec) const;
    casadi::SX euler_to_quaternion(casadi::SX& roll, casadi::SX& pitch, casadi::SX& yaw) const;
    casadi::SX euler_to_rotmatrix(casadi::SX& roll, casadi::SX& pitch, casadi::SX& yaw) const;
    casadi::SX eulerd2w(casadi::SX& euler, casadi::SX& eulerd) const;
    casadi::SX wd2eulerdd(casadi::SX& euler, casadi::SX& eulerd, casadi::SX& wd) const;
    casadi::SX load_factor(casadi::SX& euler_angles) const;
    casadi::SX twist_body_angular(casadi::SX& state) const;
    casadi::SX toBody(casadi::SX& state, casadi::SX& vec) const;
    casadi::SX get_acceleration_commands(casadi::SX& state, casadi::SX& command) const;
    casadi::SX get_torques(casadi::SX& acceleration_command, casadi::SX& state, casadi::SX& command, casadi::SX& twist_body) const;
    casadi::SX get_force(casadi::SX& state, casadi::SX acceleration_command_z) const;
    casadi::SX AdjointTwist(casadi::SX& twist) const;
    casadi::SX get_euler_accs(casadi::SX& state, casadi::SX& torques, casadi::SX& rel_force) const;
    casadi::SX get_cartesian_accs(casadi::SX& state, casadi::SX& rel_force_command) const;
    casadi::SX state_vec_from_acc(casadi::SX& state, casadi::SX& acc_vec, casadi::SX& twist_body) const;
    casadi::SX forward(casadi::SX& x, casadi::SX& u) const;*/
     
    
    // NONLINEAR SYSTEM FUNCTIONS (CasADi semantics !)
    // TEMPLATES FOR SUPPORTING BOTH SYMBOLIC AND REAL !
    template<typename T>
    T VecToso3(T vec) const
    {
        T so3 = T::zeros(3, 3);
        so3(0, 1) = -vec(2);
        so3(0, 2) =  vec(1);
        so3(1, 0) =  vec(2);
        so3(1, 2) = -vec(0);
        so3(2, 0) = -vec(1);
        so3(2, 1) =  vec(0);
        return so3;
    }

    template<typename T>
    T euler_to_quaternion(T& roll, T& pitch, T& yaw) const
    {
        T q = T::zeros(4);
        q(1) = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2);
        q(2) = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2);
        q(3) = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2);
        q(0) = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2);
        return q;
    }

    template<typename T>
    T euler_to_rotmatrix(T& roll, T& pitch, T& yaw) const
    {
        T m = T::zeros(3, 3);
        m(0, 0) = cos(yaw)*cos(pitch);
        m(0, 1) = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll);
        m(0, 2) = cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll);
        m(1, 0) = sin(yaw)*cos(pitch);
        m(1, 1) = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll);
        m(1, 2) = sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll);
        m(2, 0) = -sin(pitch);
        m(2, 1) = sin(roll)*cos(pitch);
        m(2, 2) = cos(roll)*cos(pitch);
        return m;
    }

    template<typename T>
    T eulerd2w(T& euler, T& eulerd) const
    {
        T roll(euler(0)), pitch(euler(1)), yaw(euler(2));
        T m = T::zeros(3, 3);
        m(0, 0) = 1;
        m(0, 1) = 0;
        m(0, 2) = -sin(pitch);
        m(1, 0) = 0;
        m(1, 1) = cos(roll);
        m(1, 2) = cos(pitch)*sin(roll);
        m(2, 0) = 0;
        m(2, 1) = -sin(roll);
        m(2, 2) = cos(pitch)*cos(roll);
        return mtimes(m, eulerd);
    }

    template<typename T>
    T wd2eulerdd(T& euler, T& eulerd, T& wd) const
    {
        T roll(euler(0)), pitch(euler(1)), yaw(euler(2));
        T rolld(eulerd(0)), pitchd(eulerd(1)), yawd(eulerd(2));
        //std::vector<T> m_vec = {1, 0, -sin(pitch), 0, cos(roll), cos(pitch)*sin(roll), 0, -sin(roll), cos(pitch)*cos(roll)};
        std::vector<T> minv_vec = {1, tan(pitch)*sin(roll), tan(pitch)*cos(roll), 0, cos(roll), -sin(roll), 0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)};
        std::vector<T> md_vec = {0, 0, -pitchd*cos(pitch), 0, -rolld*sin(roll), rolld*cos(pitch)*cos(roll) - pitchd*sin(roll)*sin(pitch), 0, -rolld*cos(roll), -rolld*cos(pitch)*sin(roll) - pitchd*cos(roll)*sin(pitch)};
        //T m = T::reshape(casadi::DM(m_vec), 3, 3);
        T minv = T::reshape(casadi::DM(minv_vec), 3, 3);
        T md = T::reshape(casadi::DM(md_vec), 3, 3);
        //return mtimes(inv(m), wd - mtimes(md, eulerd));
        return mtimes(minv, wd - mtimes(md, eulerd));
    }

    template<typename T>
    T load_factor(T& euler_angles) const
    {
        T roll(euler_angles(0)), pitch(euler_angles(1)), yaw(euler_angles(2));
        T q = euler_to_quaternion(roll, pitch, yaw);
        return 1 / (q(0)*q(0) - q(1)*q(1) - q(2)*q(2) + q(3)*q(3));
    }

    template<typename T>
    T twist_body_angular(T& state) const
    {
        T roll(state(9)), pitch(state(12)), yaw(state(15));
        T rolld(state(10)), pitchd(state(13)), yawd(state(16));
        T euler(horzcat(roll, horzcat(pitch, yaw))); 
        T eulerd(vertcat(rolld, vertcat(pitchd, yawd)));
        T wb = eulerd2w(euler, eulerd);
        return wb;
    }

    template<typename T>
    T toBody(T& state, T& vec) const
    {
        T m = T::zeros(3, 3);
        T roll(state(9)), pitch(state(12)), yaw(state(15));
        m(0, casadi::Slice(0, 3)) = horzcat(cos(yaw)*cos(pitch), horzcat(cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll)));
        m(1, casadi::Slice(0, 3)) = horzcat(sin(yaw)*cos(pitch), horzcat(sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll)));
        m(2, casadi::Slice(0, 3)) = horzcat(-sin(pitch), horzcat(sin(roll)*cos(pitch), cos(roll)*cos(pitch)));
        T vec_body_x = m(0, 0) * vec(0) + m(1, 0) * vec(1) + m(2, 0) * vec(2);
        T vec_body_y = m(0, 1) * vec(0) + m(1, 1) * vec(1) + m(2, 1) * vec(2);
        T vec_body_z = m(0, 2) * vec(0) + m(1, 2) * vec(1) + m(2, 2) * vec(2);
        T vec_body = horzcat(vec_body_x, horzcat(vec_body_y, vec_body_z));
        return vec_body;
    }

    template<typename T>
    T get_acceleration_commands(T& state, T& command) const
    {
        // Stating passive commands and abstract states are non zero.
        T cart_speeds[3] = {state(1), state(4), state(7)};
        T acceleration_command_x = -gamma_x * cart_speeds[0] + (1-r[0])*gamma_x*command(0) + r[0]*gamma_x*command(1);              //            /!\ Flight must be initialized !
        T acceleration_command_y = -gamma_y * cart_speeds[1] + (1-r[1])*gamma_y*command(2) + r[1]*gamma_y*command(3);
        T acceleration_command_z = -gamma_z * cart_speeds[2] + (1-r[2])*gamma_z*command(4) + r[2]*gamma_z*command(5) + g;
        return horzcat(acceleration_command_x, horzcat(acceleration_command_y, acceleration_command_z));
    }
    
    template<typename T>
    T get_acceleration_commands_debug(T& state, T& command) const
    {
        // Stating passive commands and abstract states are non zero.
        T cart_speeds[3] = {state(1), state(4), state(7)};
        T acceleration_command_x, acceleration_command_y, acceleration_command_z;
        if (!std::isnan(static_cast<double>(command(0))))
            {acceleration_command_x = -gamma_x * cart_speeds[0] + (1-r[0])*gamma_x*command(0) + r[0]*gamma_x*command(1);}              //            /!\ Flight must be initialized !
        else
            {acceleration_command_x = -gamma_x * cart_speeds[0] + gamma_x*command(1);}
        if (!std::isnan(static_cast<double>(command(2))))
            {acceleration_command_y = -gamma_y * cart_speeds[1] + (1-r[1])*gamma_y*command(2) + r[1]*gamma_y*command(3);}
        else
            {acceleration_command_y = -gamma_y * cart_speeds[1] + gamma_y*command(3);}
        if (!std::isnan(static_cast<double>(command(4))))
            {acceleration_command_z = -gamma_z * cart_speeds[2] + (1-r[2])*gamma_z*command(4) + r[2]*gamma_z*command(5) + g;}
        else
            {acceleration_command_z = -gamma_z * cart_speeds[2] + gamma_z*command(5) + g;}
        return horzcat(acceleration_command_x, horzcat(acceleration_command_y, acceleration_command_z));
    }

    template<typename T>
    T get_torques(T& acceleration_command, T& state, T& command, T& twist_body, T& rel_force) const
    {
        T acceleration_command_body = toBody(state, acceleration_command);
        T acceleration_command_body_x = acceleration_command_body(0);
        T acceleration_command_body_y = acceleration_command_body(1);
        T uwz_pass(command(6)), uwz(command(7));
        T euler  = horzcat(state(9), horzcat(state(12), state(15)));
        T eulerd = vertcat(state(10), vertcat(state(13), state(16)));
        T wz = eulerd2w(euler, eulerd)(2);
        double I1(I(0, 0)), I2(I(1, 1)), I3(I(2, 2));
        T uwx_body(-acceleration_command_body_y/g), uwy_body(acceleration_command_body_x/g);      // Induced commands
        T uwx_body_pass(command(8)), uwy_body_pass(command(9));                                   // Inner passive commands
        T tbx(twist_body(0)), tby(twist_body(1));
        T prev_tbx(state(nx-2)), prev_tby(state(nx-1));
        T lowpass_uwx_body = (1-r[3]) * uwx_body_pass + r[3] * uwx_body;
        T lowpass_uwy_body = (1-r[4]) * uwy_body_pass + r[4] * uwy_body;
        T torque_x = I1 * (gamma_wx * lowpass_uwx_body + axy_kd * ((lowpass_uwx_body - uwx_body_pass)/delta_t - (tbx - prev_tbx)));// /!\ Flight must be initialized !
        T torque_y = I2 * (gamma_wy * lowpass_uwy_body + axy_kd * ((lowpass_uwy_body - uwy_body_pass)/delta_t - (tby - prev_tby)));
        T torque_z = I3 * gamma_wz * (-wz + (1-r[5]) * uwz_pass + r[5] * uwz);
        T force_offset = T::zeros(3);
        double CoG0(CoG(0)), CoG1(CoG(1));
        force_offset(0) = CoG1*rel_force;
        force_offset(1) = -CoG0*rel_force;
        T torques = vertcat(torque_x, vertcat(torque_y, torque_z));
        torques = torques - force_offset;
        return torques;
    }
    
    template<typename T>
    T get_torques_debug(T& acceleration_command, T& state, T& command, T& twist_body, T& rel_force) const
    {
        T acceleration_command_body = toBody(state, acceleration_command);
        T acceleration_command_body_x = acceleration_command_body(0);
        T acceleration_command_body_y = acceleration_command_body(1);
        T uwz_pass(command(6)), uwz(command(7));
        T euler  = horzcat(state(9), horzcat(state(12), state(15)));
        T eulerd = vertcat(state(10), vertcat(state(13), state(16)));
        T wz = eulerd2w(euler, eulerd)(2);
        double I1(I(0, 0)), I2(I(1, 1)), I3(I(2, 2));
        T uwx_body(-acceleration_command_body_y/g), uwy_body(acceleration_command_body_x/g);      // Induced commands
        T uwx_body_pass(command(8)), uwy_body_pass(command(9));                                   // Inner passive commands
        T tbx(twist_body(0)), tby(twist_body(1));
        T prev_tbx(state(nx-2)), prev_tby(state(nx-1));
        T torque_x, torque_y, torque_z;
        T lowpass_uwx_body;
        T lowpass_uwy_body;
        if (!std::isnan(static_cast<double>(prev_tbx)) && !std::isnan(static_cast<double>(uwx_body_pass)))
            {
                lowpass_uwx_body = (1-r[3]) * uwx_body_pass + r[3] * uwx_body;
                torque_x = I1 * (gamma_wx * lowpass_uwx_body + axy_kd * ((lowpass_uwx_body - uwx_body_pass)/delta_t - (tbx - prev_tbx)));
            }  //  /!\ Flight must be initialized !
        else
            {
                lowpass_uwx_body = uwx_body;
                torque_x = I1 * (gamma_wx * uwx_body - axy_kd * tbx);
            }
        if (!std::isnan(static_cast<double>(prev_tby)) && !std::isnan(static_cast<double>(uwy_body_pass)))
            {
                lowpass_uwy_body = (1-r[4]) * uwy_body_pass + r[4] * uwy_body;
                torque_y = I2 * (gamma_wy * lowpass_uwy_body + axy_kd * ((lowpass_uwy_body - uwy_body_pass)/delta_t - (tby - prev_tby)));
            }
        else
            {
                lowpass_uwy_body = uwy_body;
                torque_y = I2 * gamma_wy * (uwy_body - axy_kd * tby);
            }
        if (!std::isnan(static_cast<double>(uwz_pass))) {torque_z = I3 * gamma_wz * (-wz + (1-r[5]) * uwz_pass + r[5] * uwz);}
        else {torque_z = I3 * gamma_wz * (-wz + uwz);}
        T force_offset = T::zeros(3);
        double CoG0(CoG(0)), CoG1(CoG(1));
        force_offset(0) = CoG1*rel_force;
        force_offset(1) = -CoG0*rel_force;
        torque_x -= force_offset(0);
        torque_y -= force_offset(1);
        T output = horzcat(torque_x, horzcat(torque_y, torque_z));
        return output;
    }

    template<typename T>
    T get_force(T& state, T acceleration_command_z) const
    {
        T euler_angles = horzcat(state(9), horzcat(state(12), state(15)));
        T forces = mass * ((acceleration_command_z - g) * load_factor(euler_angles) + g);
        return forces;
    }

    template<typename T>
    T Ad_Twist(T& twist) const
    {
        T w(horzcat(twist(0), horzcat(twist(1), twist(2)))), v(horzcat(twist(3), horzcat(twist(4), twist(5))));
        T AT = T::zeros(6, 6);
        AT(casadi::Slice(0, 3), casadi::Slice(0, 3)) = VecToso3(w);
        AT(casadi::Slice(0, 3), casadi::Slice(3, 6)) = T::zeros(3, 3);
        AT(casadi::Slice(3, 6), casadi::Slice(0, 3)) = VecToso3(v);
        AT(casadi::Slice(3, 6), casadi::Slice(3, 6)) = VecToso3(w);
        return AT;
    }
    
    template<typename T>
    T Adjoint(T& state) const
    {
        T pos = horzcat(state(0), horzcat(state(3), state(6)));
        T roll(state(9)), pitch(state(12)), yaw(state(15));
        T Ad = T::zeros(6, 6);
        T RotM = euler_to_rotmatrix(roll, pitch, yaw);
        Ad(casadi::Slice(0, 3), casadi::Slice(0, 3)) = RotM;
        Ad(casadi::Slice(0, 3), casadi::Slice(3, 6)) = T::zeros(3, 3);
        Ad(casadi::Slice(3, 6), casadi::Slice(0, 3)) = mtimes(VecToso3(pos), RotM);
        Ad(casadi::Slice(3, 6), casadi::Slice(3, 6)) = RotM;
        return Ad;
    }
    
    template<typename T>
    T AdjointInvert(T& state) const
    {
        T pos = horzcat(state(0), horzcat(state(3), state(6)));
        T roll(state(9)), pitch(state(12)), yaw(state(15));
        T AdI = T::zeros(6, 6);
        T RotM = euler_to_rotmatrix(roll, pitch, yaw);
        AdI(casadi::Slice(0, 3), casadi::Slice(0, 3)) = transpose(RotM);
        AdI(casadi::Slice(0, 3), casadi::Slice(3, 6)) = T::zeros(3, 3);
        AdI(casadi::Slice(3, 6), casadi::Slice(0, 3)) = -mtimes(transpose(RotM), VecToso3(pos));
        AdI(casadi::Slice(3, 6), casadi::Slice(3, 6)) = transpose(RotM);
        return AdI;
    }

    template<typename T>
    T get_euler_accs(T& state, T& torques) const
    {
        T euler = horzcat(state(9), horzcat(state(12), state(15)));
        T eulerd = vertcat(state(10), vertcat(state(13), state(16)));
        T roll(state(9)), pitch(state(12)), yaw(state(15));
        T RotM = euler_to_rotmatrix(roll, pitch, yaw);
        T body_twist = T::zeros(6);
        body_twist(casadi::Slice(0, 3)) = eulerd2w(euler, eulerd);
        body_twist(casadi::Slice(3, 6)) = mtimes(transpose(RotM), vertcat(state(1), vertcat(state(4), state(7))));
        T wrench = T::zeros(6);
        for (int i(0); i<3; i++) {wrench(i+3) = torques(i);}
        T G = T::zeros(6, 6);
        G(casadi::Slice(0, 3), casadi::Slice(0, 3)) = OI;
        G(casadi::Slice(0, 3), casadi::Slice(3, 6)) = mass * VecToso3(CoG);
        G(casadi::Slice(3, 6), casadi::Slice(0, 3)) = -mass * VecToso3(CoG);
        G(casadi::Slice(3, 6), casadi::Slice(3, 6)) = mass * T::eye(3);
        std::cout << "\nG matrix:\n" << std::endl;
        for (int i(0); i<6; i++)
        {
            for (int j(0); j<6; j++) {std::cout << G(i, j) << " ";}
            std::cout << std::endl;
        }
        std::cout << "\n";
        T der_body_twist = T::zeros(6);
        T Ginv = inv(G);
        std::cout << "\nGinv matrix:\n" << std::endl;
        for (int i(0); i<6; i++)
        {
            for (int j(0); j<6; j++) {std::cout << Ginv(i, j) << " ";}
            std::cout << std::endl;
        }
        std::cout << "\nwrench torques:\n" << std::endl;
        std::cout << static_cast<std::vector<double>>(wrench) << std::endl;
        
        T v1 = mtimes(G, body_twist);
        T v2 = transpose(Ad_Twist(body_twist));
        der_body_twist = mtimes(Ginv, wrench + mtimes(v2, v1));
        
        std::cout << "\ntransposed adjoint twist:\n" << std::endl;
        for (int(i); i<6; i++)
            std::cout << static_cast<std::vector<double>>(transpose(Ad_Twist(body_twist))(i, casadi::Slice())) << std::endl;
        std::cout << "\nInner der body twist:" << std::endl;
        std::cout << static_cast<std::vector<double>>(der_body_twist) << std::endl;
        std::cout << "\n";
        T d_wb = der_body_twist(casadi::Slice(0, 3));
        T all_euler_accs = wd2eulerdd(euler, eulerd, d_wb);
        return all_euler_accs;
    }

    template<typename T>
    T get_cartesian_accs(T& state, T& rel_force_command) const
    {
        T roll(state(9)), pitch(state(12)), yaw(state(15));
        T acc_x = rel_force_command/mass * (cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll));
        T acc_y = rel_force_command/mass * (sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll));
        T acc_z = rel_force_command/mass * cos(pitch)*cos(roll) - g;
        return vertcat(acc_x, vertcat(acc_y, acc_z));
    }

    template<typename T>
    T state_vec_from_acc(T& state, T& acc_vec, T& twist_body) const
    {
        T state_vec = T::zeros(nx);
        for (int i(0); i<6; i++)
        {
            state_vec(3*i)   = state(3*i) + delta_t * state(3*i+1) + delta_t*delta_t/2 * state(3*i+2);
            state_vec(3*i+1) = state(3*i+1) + delta_t * state(3*i+2);
            state_vec(3*i+2) = acc_vec(i);
        }
        state_vec(nx-2) = twist_body(0);       // Setting previous twist body
        state_vec(nx-1) = twist_body(1);
        return state_vec;
    }
    
    template<typename T>
    T forward(T x, T u, bool _debug=false)
    {
        // Getting current body twist
        T euler = vertcat(x(9), vertcat(x(12), x(15)));
        T eulerd = vertcat(x(10), vertcat(x(13), x(16)));
        T tba = eulerd2w(euler, eulerd);

        T acc_commands;
        T relative_z_force;
        T torques;
        if (!_debug) 
        {
            acc_commands = get_acceleration_commands(x, u);
            T acc_command_z = acc_commands(2);
            relative_z_force = get_force(x, acc_command_z);
            torques = get_torques(acc_commands, x, u, tba, relative_z_force);
        }
        else 
        {
            acc_commands = get_acceleration_commands_debug(x, u);
            T acc_command_z = acc_commands(2);
            relative_z_force = get_force(x, acc_command_z);
            torques = get_torques_debug(acc_commands, x, u, tba, relative_z_force);
        }
        T output_cart_acc = get_cartesian_accs(x, relative_z_force);
        T output_euler_acc = get_euler_accs(x, torques);
        T output_acc = vertcat(output_cart_acc, output_euler_acc);
        T output = state_vec_from_acc(x, output_acc, tba);
        
        if (_debug)
        {
            std::cout << "accelerations commands :" << static_cast<std::vector<double>>(acc_commands) << std::endl;
            std::cout << "relative z force :" << static_cast<std::vector<double>>(relative_z_force) << std::endl;
            std::cout << "accelerations :" << static_cast<std::vector<double>>(output_acc) << std::endl;
            std::cout << "output: "  << static_cast<std::vector<double>>(output) << std::endl;
        }
        
        return output;
    }

    // END OF NONLINEAR SYSTEM FUNCTIONS
    
    
    void debug(bool initialization=true);
    void build_solver();
    void build_linear_solver();
    void iteration_test();
    void run_trajectory(const ros::TimerEvent&);
    
    
    
    

protected:
    void command_callback(const geometry_msgs::Twist::ConstPtr& twist);
    void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose);
    void euler_callback(const geometry_msgs::PoseStamped::ConstPtr& pose);
    void traj_length_callback(const std_msgs::Int16::ConstPtr& msg);
    void traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void timestep_callback(const std_msgs::Float32::ConstPtr& dt);
    void ui_command_callback(const std_msgs::Int16::ConstPtr& msg);
    void init_parameters();
    void constraints_matrices();
    //static double cost(const Eigen::VectorXd& c, const Eigen::VectorXd& x, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref);
    
    
    
    
    
private:
    bool ground_truth;
    bool wind;
    bool debug_mode=false;
    double mass=1.478;
    casadi::DM I, OI, CoG;
    std::string method;
    int trajL=0; 
    int N, linear_N;  // Horizons
    int nx=20; 
    int nu=10;
    int linear_nx=18;
    int linear_nu=4;
    int step = 0;
    double P_factor;
    std::vector<double> sol_prev;
    casadi::Function solver;
    casadi::Function linear_solver;
    std::map<std::string, casadi::DM> args;
    casadi::DM x0, u0;
    casadi::DM u_opt;
    std_msgs::Float64MultiArray result_traj;
    geometry_msgs::Twist input;
    Eigen::Vector3d current_pose, current_vel, current_acc;
    Eigen::Vector3d current_euler, current_euler_vel, current_euler_acc;
    Eigen::Vector4d current_ori, current_command;
    Eigen::Vector3d current_body_twist_w;
    casadi::DM xref, uref, linear_xref, linear_uref;
    casadi::DM xref_vec, xref_vec_final, linear_xref_vec, linear_xref_vec_final;
    double delta_t;    // controller chosen frequency
    double dt=0.01;    // controller internal clock
    double g=9.8065;
    double lxy_kp, lxy_ki, lxy_kd, lz_kp, lz_ki, lz_kd;
    double axy_kp, axy_ki, axy_kd, az_kp, az_ki, az_kd;
    double lxy_tau, lz_tau, axy_tau, az_tau;
    double gamma_x, gamma_y, gamma_z, gamma_wx, gamma_wy, gamma_wz;
    double r[6];
    std::vector<double> xmin, xmax;
    std::vector<double> umin, umax;
    std::vector<Eigen::Vector3d> traj_points, traj_speeds, traj_accs;
    std::vector<double> traj_yaws, traj_yaws_speeds, traj_yaws_accs;
    ros::NodeHandle _nh;
    ros::Subscriber pose_subscriber, euler_subscriber, command_subscriber;
    ros::Subscriber traj_length_subscriber, traj_points_subscriber, traj_speeds_subscriber, traj_accs_subscriber;
    ros::Subscriber yaws_subscriber, yaws_speeds_subscriber, yaws_accs_subscriber;
    ros::Subscriber timestep_subscriber, ui_command_subscriber;
    ros::Publisher exe_pub, result_traj_pub;
    std::shared_ptr<ros::Timer> timer;
    std::vector<double> yaws;
    std::vector<double> yaws_speeds;
    std::vector<double> yaws_accs;
    ros::Subscriber traj_sub, speeds_sub, accs_sub, yaws_sub, yaws_speeds_sub, yaws_accs_sub;
};

#endif // LINEAR_MPC_H_INCLUDED
