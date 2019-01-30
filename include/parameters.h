#ifndef dealii__cppls_parameters_h
#define dealii__cppls_parameters_h

#include <deal.II/base/parameter_handler.h>

#include <string>

namespace CPPLS
{
using namespace dealii;

class Parameters
{
public:
    bool compute_temperature;
    bool linear_in_void_ratio;
    bool use_advance;

    unsigned int dimension;
    double base_sedimentation_rate;
    double box_size;
    double x_length;
    double y_length;
    double z_length;
    bool cubic;
    unsigned int n_layers;

    unsigned int initial_refinement_level;
    unsigned int degree;
    unsigned int degree_LS;

    double stop_time;
    double theta;
    double cfl;
    unsigned int n_reps;

    double nl_tol;
    unsigned int maxiter;

    double sigma_tol;

    unsigned int output_interval;

    void read_parameter_file(const std::string &file_name);
private:
    void configure_parameter_handler(ParameterHandler &parameter_handler);
};
}
#endif
