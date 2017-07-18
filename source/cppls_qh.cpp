

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
    using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

//#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
//#include <deal.II/numerics/solution_transfer.h>
//#include <deal.II/numerics/matrix_tools.h>


#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <iostream>

#include "my_utility_functions.h"
#include "parameters.h"


namespace CPPLS
{
	using namespace dealii;

	template <int dim>
	class LayerMovementProblem
	{
	public:
		LayerMovementProblem (const CPPLS::Parameters &parameters);
		~LayerMovementProblem ();
		void run();

	private:
	//Member Data	
		//runtime parameters
		const CPPLS::Parameters parameters;
		//mpi communication
  		MPI_Comm mpi_communicator;
		const unsigned int n_mpi_processes;
		const unsigned int this_mpi_process;
		//mesh
		parallel::distributed::Triangulation<dim> triangulation;
		//FE basis spaces
		//pressure
		int                  degree_P;
	  	DoFHandler<dim>      dof_handler_P;
	  	FE_Q<dim>            fe_P;
	  	IndexSet             locally_owned_dofs_P;
	  	IndexSet             locally_relevant_dofs_P;
	  	//temperature
	  	int                  degree_T;
	  	DoFHandler<dim>      dof_handler_T;
	  	FE_Q<dim>            fe_T;
	  	IndexSet             locally_owned_dofs_T;
	  	IndexSet             locally_relevant_dofs_T;
	  	// level set (can multiple level sets use the same of below? probably not IndexSets)std::vector<IndexSets>
	  	int                  degree_LS;
	  	DoFHandler<dim>      dof_handler_LS;
	  	FE_Q<dim>            fe_LS;
	  	IndexSet             locally_owned_dofs_LS;
	  	IndexSet             locally_relevant_dofs_LS;

	  	//output stream where only mpi rank 0 output gets to stdout
	  	ConditionalOStream                pcout;


		const double time_step;
  		double current_time;

  		//FE Field Solution Vectors

		  LA::MPI::Vector locally_relevant_solution_u; //ls
		  LA::MPI::Vector locally_relevant_solution_p;
		  LA::MPI::Vector locally_relevant_solution_t;
		  LA::MPI::Vector locally_relevant_solution_f; //speed function
		  LA::MPI::Vector completely_distributed_solution_u;
		  LA::MPI::Vector completely_distributed_solution_p;
		  LA::MPI::Vector completely_distributed_solution_t;
		  LA::MPI::Vector completely_distributed_solution_f;

		  LA::MPI::Vector overburden;
		  LA::MPI::Vector bulkdensity;
		  LA::MPI::Vector porosity;


  		// for boundary conditions
	  	std::vector<unsigned int> boundary_values_id_u;
	  	std::vector<unsigned int> boundary_values_id_p;
	  	std::vector<unsigned int> boundary_values_id_t;
	  	std::vector<double> boundary_values_u;
	  	std::vector<double> boundary_values_p;
	  	std::vector<double> boundary_values_t;


  		//Physical Vectors




  	//Member Functions

	  //create mesh
	  void setup_geometry();

	  void setup_material_configuration(LA::MPI::Vector/*std::vector<LA::MPI::Vector>*/);
	  // initialize vectors
	  void setup_dofs_P();
	  void setup_dofs_T();

	  void assemble_system_P();
	  void assemble_system_T();
		
	  void compute_bulkdensity();
	  void compute_overburden();
	  void compute_porosity();
	  void compute_speed_function();

	  void output_vectors();

	};





//Constructor

	template<int dim>
 	LayerMovementProblem<dim>::LayerMovementProblem (const CPPLS::Parameters &parameters)
 	:
	parameters(parameters),
	mpi_communicator (MPI_COMM_WORLD),
 	n_mpi_processes {Utilities::MPI::n_mpi_processes(mpi_communicator)},
 	this_mpi_process {Utilities::MPI::this_mpi_process(mpi_communicator)},
 	{};


//	
template<int dim>
void
	LayerMovementProblem<dim>::setup_geometry()
	{
		  //GridGenerator::hyper_cube(triangulation, 0, parameters.box_size);
		  //GridGenerator::subdivided_hyper_rectangle(triangulation, 0, parameters.box_size);
   		  triangulation.refine_global(parameters.initial_refinement_level);

	}


template <int dim>
	void
	LayerMovementProblem<dim>::compute_bulkdensity()
	{
		for (auto cell : filter_iterators(triangulation.active_cell_iterators(),
													IteratorFilters::LocallyOwnedCell()))
		  {
		    fe_values.reinit (cell);
		    //query the local porosity
		    
		  } 
	}


template <int dim>
	void
	LayerMovementProblem<dim>::compute_overburden()
	{

	    const QGauss<dim>  quadrature_formula(3);

	    FEValues<dim> fe_values (fe, quadrature_formula,
	                             update_values    |  update_gradients |
	                             update_quadrature_points |
	                             update_JxW_values);

	    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	    const unsigned int   n_q_points    = quadrature_formula.size();

	    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	    Vector<double>       cell_rhs (dofs_per_cell);

	    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		for (auto cell : filter_iterators(triangulation.active_cell_iterators(),
													IteratorFilters::LocallyOwnedCell())
		  {
		    fe_values.reinit (cell);

		 
		  } 
	}


template <int dim>
	void LayerMovementProblem<dim>::assemble_system_P()
	{

    TimerOutput::Scope t(computing_timer, "assembly_P");
     const QGauss<dim>  quadrature_formula(3);

     // RightHandSide<dim> right_hand_side;
     // //set time too
     // right_hand_side.set_time(time);
     // DiffusionCoefficient<dim> diffusion_coeff;


    FEValues<dim> fe_values (fe_P, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_laplace_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   cell_mass_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double> u_at_quad(n_q_points);


    // typename DoFHandler<dim>::active_cell_iterator
    // cell = dof_handler.begin_active(),
    // endc = dof_handler.end();
    // for (; cell!=endc; ++cell)
    //   if (cell->is_locally_owned())
    //     {

      for (auto cell : filter_iterators(dof_handler.active_cell_iterators(),
										IteratorFilters::LocallyOwnedCell())
  		{ 
          cell_laplace_matrix = 0;
          cell_mass_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit (cell);
          fe_values.get_function_values(interface_phi,phi_at_quad);
          fe_values.get_function_values(locally_relevant_solution_p,pressure_at_quad);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
              poro = get_poro(phi_at_quad[q_point]);
	          press = get_pressure(pressure_at_quad[q_point]);
              poro = exp(-k*press);
               
            }

          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (cell_laplace_matrix,
                                                  cell_rhs,
                                                  local_dof_indices,
                                                  laplace_matrix,
                                                  forcing);
          constraints.distribute_local_to_global (cell_mass_matrix,
                                                  local_dof_indices,
                                                  mass_matrix);

        }

    // Notice that the assembling above is just a local operation. So, to
    // form the "global" linear system, a synchronization between all
    // processors is needed. This could be done by invoking the function
    // compress(). See @ref GlossCompress  "Compressing distributed objects"
    // for more information on what is compress() designed to do.
    laplace_matrix.compress (VectorOperation::add);
    mass_matrix.compress (VectorOperation::add);
    forcing.compress (VectorOperation::add);


  }

		

	}



template <int dim>
void 
LayerMovementProblem<dim>::output_vectors()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_u, "u");
  data_out.build_patches ();
  
  const std::string filename = ("sol_vectors-" +
				Utilities::int_to_string (output_number, 3) +
				"." +
				Utilities::int_to_string
				(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);
  
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;
	   i<Utilities::MPI::n_mpi_processes(mpi_communicator);
	   ++i)
	filenames.push_back ("sol_vectors-" +
			     Utilities::int_to_string (output_number, 3) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}








template<int dim>
	LayerMovementProblem<dim>::run()
	{
		//common mesh
		setup_geometry();
		// 
		//initialize level set solver
		LevelSetSolver<dim> level_set_solver;

		//initialize pressure solver
		PressureEquation<dim> pressure_solver;
		//initialize temperature solver
		//TemperatureEquation<dim> temperature_solver;





  // TIME STEPPING
  for (timestep_number=1, time=time_step; time<=final_time;
       time+=time_step,++timestep_number)
    {
      pcout << "Time step " << timestep_number 
	    << " at t=" << time 
	    << std::endl;
      // GET NAVIER STOKES VELOCITY
      navier_stokes.set_phi(locally_relevant_solution_phi);
      navier_stokes.nth_time_step(); 
      navier_stokes.get_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
      transport_solver.set_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
      // GET LEVEL SET SOLUTION
      transport_solver.nth_time_step();
      transport_solver.get_unp1(locally_relevant_solution_phi);      
      if (get_output && time-(output_number)*output_time>0)
	output_results();
    }




	}



} //end namespace CPPLS

constexpr int dim {3};


int main(int argc, char *argv[])
{
  // One of the new features in C++11 is the <code>chrono</code> component of
  // the standard library. This gives us an easy way to time the output.
  try
    {
	    using namespace dealii;
	    using namespace CPPLS;
	      
	    auto t0 = std::chrono::high_resolution_clock::now();

	  	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	  	
	  	CPPLS::Parameters parameters;
	  	parameters.read_parameter_file("parameters.prm");
	  	
	  	LayerMovementProblem<dim> run_layers(parameters);
	  	run_layers.run();

	  	auto t1 = std::chrono::high_resolution_clock::now();
	  	if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    {
	      std::cout << "time elapsed: "
	                << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
	                << " milliseconds."
	                << std::endl;
	    }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
  
}
