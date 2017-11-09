#include "parameters.h"

#include <fstream>
#include <string>

namespace CPPLS
{
void Parameters::configure_parameter_handler(ParameterHandler &parameter_handler)
{
    parameter_handler.enter_subsection("Geometry");
    {
        parameter_handler.declare_entry
        ("dimension", "2",Patterns::Integer(2,3), "dimension of problem" );
        parameter_handler.declare_entry
        ("base_sedimentation_rate", "3.147e-11", Patterns::Double(0, 1), "base sedimentation rate");
        parameter_handler.declare_entry
        ("box_size", "1000",Patterns::Integer(2,1000000), "size of square domain" );

    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Layer Parameters");
    {
        parameter_handler.declare_entry
        ("n_layers", "1", Patterns::Integer (0,100), "Number of sediment layers");
        //parameter_handler.declare_entry
        //("layer_order", "1.0", Patterns::List() , "Sequence of rock types");
        parameter_handler.declare_entry
        ("time_dependent_forcing", "true", Patterns::Bool(), "Whether or not "
         "the forcing function depends on time.");
    }
    parameter_handler.leave_subsection();


    parameter_handler.enter_subsection("Finite Element");
    {

        parameter_handler.declare_entry
        ("initial_refinement_level", "1", Patterns::Integer(1),
         "Initial number of levels in the mesh.");
        parameter_handler.declare_entry
        ("degree", "1", Patterns::Integer(1),
         "Finite element order of physical quantity dof handler");
        parameter_handler.declare_entry
        ("degree_LS", "1", Patterns::Integer(1), "Finite element order of Level set solver.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Time Step");
    {

        parameter_handler.declare_entry
        ("start_time", "0.0", Patterns::Double(0.0), "Start time.");
        parameter_handler.declare_entry
        ("stop_time", "1.0", Patterns::Double(1.0), "Stop time.");
        parameter_handler.declare_entry
        ("n_time_steps", "1", Patterns::Integer(1), "Number of time steps.");
        parameter_handler.declare_entry
        ("theta", "0.5",Patterns::Double(0.5), "Theta-Scheme value 0 explicit, 1/2 CN, 1 implicit" );
        parameter_handler.declare_entry
        ("cfl","0.5", Patterns::Double(0.5), "cfl condition constant for time step, both X and LS");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {

        parameter_handler.declare_entry
        ("output_interval", "10", Patterns::Integer(1), "Output interval.");
        parameter_handler.declare_entry
        ("patch_level", "2", Patterns::Integer(0), "Patch level.");
    }
    parameter_handler.leave_subsection();
}

void Parameters::read_parameter_file(const std::string &file_name)
{
    ParameterHandler parameter_handler;
    {
        std::ifstream file(file_name);
        configure_parameter_handler(parameter_handler);
        parameter_handler.parse_input(file);
    }

    parameter_handler.enter_subsection("Geometry");
    {
        dimension = parameter_handler.get_integer("dimension");
        box_size = parameter_handler.get_double("box_size");
        base_sedimentation_rate = parameter_handler.get_double("base_sedimentation_rate");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Layer Parameters");
    {
        n_layers = parameter_handler.get_double("n_layers");

        time_dependent_forcing = parameter_handler.get_bool("time_dependent_forcing");
    }
    parameter_handler.leave_subsection();


    parameter_handler.enter_subsection("Finite Element");
    {

        initial_refinement_level = parameter_handler.get_integer("initial_refinement_level");
        degree = parameter_handler.get_integer("degree");
        degree_LS = parameter_handler.get_integer("degree_LS");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Time Step");
    {

        start_time = parameter_handler.get_double("start_time");
        stop_time = parameter_handler.get_double("stop_time");
        n_time_steps = parameter_handler.get_integer("n_time_steps");
        theta = parameter_handler.get_double("theta");
        cfl = parameter_handler.get_double("cfl");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {

        output_interval = parameter_handler.get_integer("output_interval");
        patch_level = parameter_handler.get_integer("patch_level");
    }
    parameter_handler.leave_subsection();
}
}
