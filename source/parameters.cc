#include "parameters.h"

#include <fstream>
#include <string>

namespace CPPLS
{
void Parameters::configure_parameter_handler(ParameterHandler &parameter_handler)
{
  parameter_handler.enter_subsection("General");
  {
    parameter_handler.declare_entry
    ("compute_temperature", "true", Patterns::Bool(), "Whether or not "
     "to calculate the diffusive temperature field");
    parameter_handler.declare_entry
    ("linear_in_void_ratio", "true", Patterns::Bool(), "If true use a compaction law "
     "that is linear in the void ratio. If false default to Athy's law");
    parameter_handler.declare_entry
    ("use_advance", "true", Patterns::Bool(), "If true use the advance_old_vectors routine");



  }
    parameter_handler.leave_subsection();


    parameter_handler.enter_subsection("Geometry");
    {
        parameter_handler.declare_entry
        ("dimension", "2",Patterns::Integer(2,3), "dimension of problem" );
        parameter_handler.declare_entry
        ("base_sedimentation_rate", "3.147e-11", Patterns::Double(0, 1), "base sedimentation rate");
        parameter_handler.declare_entry
        ("box_size", "1000",Patterns::Double(2,1000000), "size of square domain" );
        parameter_handler.declare_entry
        ("x_length", "1000",Patterns::Double(2,1000000), "size of x length" );
        parameter_handler.declare_entry
        ("y_length", "1000",Patterns::Double(2,1000000), "size of y length" );
        parameter_handler.declare_entry
        ("z_length", "1000",Patterns::Double(2,1000000), "size of z length" );
        parameter_handler.declare_entry
        ("cubic", "true", Patterns::Bool(), "Whether or not "
         "domain is a cube/square");


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
        ("degree", "1", Patterns::Integer(1,5),
         "Finite element order of physical quantity dof handler");
        parameter_handler.declare_entry
        ("degree_LS", "1", Patterns::Integer(1,5), "Finite element order of Level set solver.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Time Step");
    {
        parameter_handler.declare_entry
        ("stop_time", "1.0", Patterns::Double(1.0), "Stop time.");

        parameter_handler.declare_entry
        ("theta", "0.5",Patterns::Double(0,1), "Theta-Scheme value 0 explicit, 1/2 CN, 1 implicit" );
        parameter_handler.declare_entry
        ("cfl","0.5", Patterns::Double(0,1), "cfl condition constant for time step, both X and LS");
        parameter_handler.declare_entry
        ("n_reps","1", Patterns::Integer(1,100),
         "the number of times the LS time stepping is run each physical time step");
    }
    parameter_handler.leave_subsection();



    parameter_handler.enter_subsection("Nonlinear Solver Picard");
    {
      parameter_handler.declare_entry
      ("nl_tol","1e-8", Patterns::Double(0,1), "nonlinear loop convergence tolerance");
      parameter_handler.declare_entry
      ("maxiter","20", Patterns::Integer(1,300), "maximum nonlinear iterations");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Linear Solver Tolerances");
    {
      parameter_handler.declare_entry
      ("sigma_tol","1e-6", Patterns::Double(0,1), "overburden solver tolerance");

    }
    parameter_handler.leave_subsection();



    parameter_handler.enter_subsection("Output");
    {

        parameter_handler.declare_entry
        ("output_interval", "10", Patterns::Integer(1), "Output interval.");

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
    parameter_handler.enter_subsection("General");
    {
      compute_temperature = parameter_handler.get_bool("compute_temperature");
      linear_in_void_ratio = parameter_handler.get_bool("linear_in_void_ratio");
      use_advance = parameter_handler.get_bool("use_advance");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Geometry");
    {
        dimension = parameter_handler.get_integer("dimension");
        base_sedimentation_rate = parameter_handler.get_double("base_sedimentation_rate");
        box_size = parameter_handler.get_double("box_size");
        x_length = parameter_handler.get_double("x_length");
        y_length = parameter_handler.get_double("y_length");
        z_length = parameter_handler.get_double("z_length");
        cubic = parameter_handler.get_bool("cubic");

    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Layer Parameters");
    {
        n_layers = parameter_handler.get_integer("n_layers");
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
        stop_time = parameter_handler.get_double("stop_time");
        theta = parameter_handler.get_double("theta");
        cfl = parameter_handler.get_double("cfl");
        n_reps = parameter_handler.get_integer("n_reps");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Nonlinear Solver Picard");
    {
      nl_tol=parameter_handler.get_double("nl_tol");
      maxiter= parameter_handler.get_integer("maxiter");

    }
    parameter_handler.leave_subsection();


    parameter_handler.enter_subsection("Linear Solver Tolerances");
    {
      sigma_tol=parameter_handler.get_double("sigma_tol");

    }
    parameter_handler.leave_subsection();


    parameter_handler.enter_subsection("Output");
    {

        output_interval = parameter_handler.get_integer("output_interval");

    }
    parameter_handler.leave_subsection();
}
}
