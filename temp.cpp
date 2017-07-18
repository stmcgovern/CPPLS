


namespace CPPLS
{
    using namespace dealii;



template <int dim>
class LayerMovementProblem 
{
public:
    class Parameters
    {
    public:
        Parameters ();
        static void declare_parameters (ParameterHandler &prm);
        void get_parameters (ParameterHandler &prm);

        unsigned int n_layers;
        unsigned int n_refinement_cycles;
        unsigned int fe_degree;

        double convergence_tolerance;
    };

    LayerMovementProblem (const Parameters &parameters);
    ~LayerMovementProblem ();

    void run();

private:
    void initialize_problem();
    
    const Parameters &parameters;
    const RockType rock_type;
    FE_Q<dim>        fe;
    
};

template <int dim>
LayerMovementProblem<dim>::Parameters::Parameters ()
    :
    n_layers (2),
    n_refinement_cycles(4),
    fe_degree (2),
    convergence_tolerance(1e-12)
{}


template <int dim>
void
LayerMovementProblem<dim>::Parameters::
declare_parameters (Para &prm)
{
    prm.declare_entry ("Number of layers", "2",
                        Patterns::Integer (),
                        "The number of layers used");
    prm.declare_entry ("Refinement cycles", "5",
                       Patterns::Integer (),
                       "Number of refinement cycles to be performed");
    prm.declare_entry ("Finite element degree", "2",
                       Patterns::Integer (),
                       "Polynomial degree of the finite element to be used");
    prm.declare_entry ("Iteration tolerance", "1e-12",
                       Patterns::Double (),
                       "Iterate something"); //TODO: better description
}

template <int dim>
void
LayerMovementProblem<dim>::Parameters::
get_parameters (Para &prm)
{
    n_layers  = prm.get_integer ("Number of layers");
    n_refinement_cycles = prm.get_integer ("Refinement cycles");
    fe_degree = prm.get_integer("Finite element degree");
    convergence_tolerance = prm.get_double ("Iteration tolerance");
}


template  <int dim>
LayerMovementProblem<dim>::
LayerMovementProblem (const Parameters &parameters)
    :
    parameters (parameters),
    rock_type (parameters.n_layers),
    fe (parameters.fe_degree)
{}

template <int dim>
LayerMovementProblem<dim>::
~LayerMovementProblem()
{
    pass
}


template <int dim>
void
LayerMovementProblem<dim>::
run()
{
    //
}




}//namespace 






int main (int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace CPPLS;

      //the last argument is the threads per core. set to negative to have tbb auto set it
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);



      std::string filename;
      if (argc < 2)
        filename = "project.prm";
      else
        filename = argv[1];
      const unsigned int dim = 2;
      ParameterHandler parameter_handler;
      LayerMovementProblem<dim>::Parameters parameters;
      parameters.declare_parameters (parameter_handler);
      parameter_handler.read_input (filename);
      parameters.get_parameters (parameter_handler);
      LayerMovementProblem<dim> layer_movement_problem (parameters);
      layer_movement_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}



