#ifndef dealii__cppls_parameters_h
#define dealii__cppls_parameters_h

#include <deal.II/base/parameter_handler.h>

#include <string>

// I prefer to use the ParameterHandler class in a slightly different way than
// usual: The class Parameters creates, uses, and then destroys a
// ParameterHandler inside the <code>read_parameter_file</code> method instead
// of keeping it around. This is nice because now all of the run time
// parameters are contained in a simple class and it can be copied or passed
// around very easily.
namespace CPPLS
{
using namespace dealii;

class Parameters
{
public:
    bool compute_temperature;

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

    double start_time;
    double stop_time;
    double theta;
    double cfl;
    double nl_tol;
    unsigned int maxiter;


    unsigned int output_interval;

    void read_parameter_file(const std::string &file_name);
private:
    void configure_parameter_handler(ParameterHandler &parameter_handler);
};
}
#endif
