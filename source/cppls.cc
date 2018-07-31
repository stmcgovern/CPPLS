

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA {
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
} // namespace LA

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
//#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
//#include <deal.II/lac/petsc_precondition.h>

//#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
//#include <deal.II/numerics/solution_transfer.h>
//#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <iostream>
#include <functional>

#include "level_set_solver.h"
#include "material_data.h"
#include "my_utility_functions.h"
#include "parameters.h"
#include "physical_functions.h"

namespace CPPLS {
using namespace dealii;


template <int dim>
class LayerMovementProblem {
public:
    LayerMovementProblem(const CPPLS::Parameters& parameters, const CPPLS::MaterialData& material_data);
    ~LayerMovementProblem();
    void run();

private:

    // Member Data
    // runtime parameters
    const CPPLS::Parameters parameters;
    const CPPLS::MaterialData material_data;

    // mpi communication
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    // mesh
    parallel::distributed::Triangulation<dim> triangulation;

    // FE basis space (for P,T, F, and sigma)
    // LS separate

    // pressure
    int degree;
    DoFHandler<dim> dof_handler;
    FE_Q<dim> fe;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;


    int degree_LS;
    DoFHandler<dim> dof_handler_LS;
    FE_Q<dim> fe_LS;
    IndexSet locally_owned_dofs_LS;
    IndexSet locally_relevant_dofs_LS;

    // output stream where only mpi rank 0 output gets to stdout
    ConditionalOStream pcout;

    TimerOutput computing_timer;

    double time_step;
    double current_time;
    double output_number;
    double final_time;
    int timestep_number;
    int out_index;

    // set timestepping scheme 1 implicit euler, 1/2 CN, 0 explicit euler
    const double theta;

    ConstraintMatrix constraints_P;
    ConstraintMatrix constraints_T;
    ConstraintMatrix constraints_LS;
    ConstraintMatrix constraints_F;
    ConstraintMatrix constraints_Sigma;

    // FE Field Solution Vectors
    // Ghosted
    // LS
    LA::MPI::Vector locally_relevant_solution_LS_0; // ls
    LA::MPI::Vector old_locally_relevant_solution_LS_0;

    // Pressure
    LA::MPI::Vector locally_relevant_solution_P;
    LA::MPI::Vector old_locally_relevant_solution_P;
    // for use in nonlinear iteration
    LA::MPI::Vector temp_locally_relevant_solution_P;
    LA::MPI::Vector old_temp_locally_relevant_solution_P;

    // Temperature

    LA::MPI::Vector locally_relevant_solution_T;
    LA::MPI::Vector old_locally_relevant_solution_T;

    // Speed function
    LA::MPI::Vector locally_relevant_solution_F;

    // this is 0 now
    LA::MPI::Vector locally_relevant_solution_Wxy;

    // Overburden
    LA::MPI::Vector locally_relevant_solution_Sigma;
    LA::MPI::Vector old_locally_relevant_solution_Sigma;
    LA::MPI::Vector temp_locally_relevant_solution_Sigma;

    // Non-ghosted
    LA::MPI::Vector completely_distributed_solution_LS_0;
    LA::MPI::Vector completely_distributed_solution_P;
    LA::MPI::Vector completely_distributed_solution_T;
    LA::MPI::Vector completely_distributed_solution_F;
    // LA::MPI::Vector completely_distributed_solution_Sigma;

    LA::MPI::Vector rhs_P;
    LA::MPI::Vector old_rhs_P;
    LA::MPI::Vector system_rhs_P;

    LA::MPI::Vector rhs_T;
    LA::MPI::Vector old_rhs_T;
    LA::MPI::Vector system_rhs_T;

    LA::MPI::Vector rhs_Sigma;

    LA::MPI::Vector rhs_F;

    // Sparse Matrices
    LA::MPI::SparseMatrix laplace_matrix_P;
    LA::MPI::SparseMatrix mass_matrix_P;
    LA::MPI::SparseMatrix system_matrix_P;

    LA::MPI::SparseMatrix laplace_matrix_T;
    LA::MPI::SparseMatrix mass_matrix_T;
    LA::MPI::SparseMatrix system_matrix_T;

    LA::MPI::SparseMatrix system_matrix_F;

    LA::MPI::SparseMatrix system_matrix_Sigma;

    // for LS boundary conditions
    std::vector<unsigned int> boundary_values_id_LS;
    std::vector<double> boundary_values_LS;

    std::vector<std::unique_ptr<LevelSetSolver<dim>>> layers;
    std::vector<std::unique_ptr<LA::MPI::Vector>> layers_solutions;
    int n_layers;


    // store functions for porosity, permeability and compressibility
    std::function
      <const double(const double pressure, const double overburden, const double initial_porosity,
      const double compaction_coefficient, const double hydrostatic, const unsigned int material_id)>
        porosity;

    std::function
      <const double( const double current_porosity, const double compaction_coefficient, const unsigned int material_id)>
        compressibility;

//    std::function
//      <const double(const double pressure, const double overburden, const double initial_porosity,
//      const double compaction_coefficient, const double hydrostatic)>
//        permeability;





    // Member Functions


    void set_physical_functions();

    // create mesh
    void setup_geometry();

    // create fe space
    void setup_dofs();

    // create appropriately sized vectors and matrices
    void setup_system_P();
    void setup_system_T();
    void setup_system_LS();
    void setup_system_F();
    void setup_system_Sigma();

    void initial_conditions();
    void set_boundary_inlet();
    void get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
                                std::vector<double>& boundary_values_LS);

    // use level set values to set cell->material_id
    void setup_material_configuration();

    // Pressure
    void assemble_matrices_P();
    void forge_system_P();
    void solve_time_step_P();
    // Temperature
    void assemble_matrices_T();
    void forge_system_T();
    void solve_time_step_T();

    // symbol used for overburden is Sigma
    void assemble_Sigma();
    void solve_Sigma();

    // Speed function (scalar)
    void assemble_F();
    void solve_F();

    bool estimate_nl_error();
    int active_layers_in_time(double time);

    void prepare_next_time_step();

    void output_vectors_LS();
    void output_vectors();
    void output_results_pp();

    void display_vectors()
    {
        output_vectors_LS();
        output_vectors();
        output_results_pp();
        output_number++;
    }


    class Postprocessor;

};

// Constructor

template <int dim>
LayerMovementProblem<dim>::LayerMovementProblem(const CPPLS::Parameters& parameters,
        const CPPLS::MaterialData& material_data)
    : parameters(parameters)
    , material_data(material_data)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes {Utilities::MPI::n_mpi_processes(mpi_communicator)}
, this_mpi_process {Utilities::MPI::this_mpi_process(mpi_communicator)}
, triangulation(mpi_communicator,
                typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening))
, degree(parameters.degree)
, degree_LS(parameters.degree_LS)
, fe(degree)
, fe_LS(degree_LS)
, dof_handler(triangulation)
, dof_handler_LS(triangulation)
, pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
, computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
, time_step{0}
, current_time {0}
, final_time {parameters.stop_time}
, output_number {0}
, out_index{0}
, theta(parameters.theta)
{};

// Destructor
template <int dim>
LayerMovementProblem<dim>::~LayerMovementProblem()
{
    dof_handler.clear();
    dof_handler_LS.clear();
    triangulation.clear();
}





template <int dim>
void LayerMovementProblem<dim>::set_physical_functions()
{
  //The purpose of this method is to set, according to the parameter file what compaction rule
  //and associated compressibility, derivative with respect to VES, to use in the assembly methods
  pcout<<"  "<<parameters.linear_in_void_ratio<<std::endl;
  if(parameters.linear_in_void_ratio)
  {
      pcout<<"Using Linear in Void Ratio Compaction Law"<<std::endl;
      porosity= [](const double pressure, const double overburden, const double initial_porosity,
                   const double compaction_coefficient, const double hydrostatic, const unsigned int material_id)
                  {
                    if(material_id ==0) {return initial_porosity;}
                    //below is LINEAR IN VOID RATIO
                     const double init_void_ratio = initial_porosity/(1-initial_porosity);
                     const double computed_void_ratio = init_void_ratio - compaction_coefficient*(overburden - pressure - hydrostatic);

                     // Assert(init_void_ratio >= computed_void_ratio, ExcInternalError());
                     return (computed_void_ratio/(1.+computed_void_ratio));

                   };

      compressibility = []( const double current_porosity, const double compaction_coefficient, const unsigned int material_id)
         {
          if(material_id ==0) {return 0.;}
           return ((1-current_porosity)*(1-current_porosity) * compaction_coefficient);
          };
  }

  //Here we default to Athy's law
  else
  {
      pcout<<"Using Athy's Compaction Law"<<std::endl;
      porosity= [](const double pressure, const double overburden, const double initial_porosity,
          const double compaction_coefficient, const double hydrostatic, const double material_id)
         {
          if(material_id ==0) {return initial_porosity;}
          Assert(overburden - pressure - hydrostatic >= 0, ExcInternalError());

           return (initial_porosity *
                   std::exp(-1 * compaction_coefficient * (overburden - pressure - hydrostatic)));
          };

      compressibility = []( const double current_porosity, const double compaction_coefficient, const unsigned int material_id)
         {
          if(material_id ==0) {return 0.;}
           return (current_porosity * compaction_coefficient);
          };
  }
  //Permeability as a function of porosity:
  // We leave the current implementation of the linear in porosity rule hard-coded.
  //One could here put in other relationships, e.g. Kozeny-Carman, following the approach above
  //allowing for selection from the input parameter file

}



//
template <int dim>
void LayerMovementProblem<dim>::setup_geometry()
{
    TimerOutput::Scope t(computing_timer, "setup_geometry");
    if (parameters.cubic)
    {
        GridGenerator::hyper_cube(triangulation, 0, parameters.box_size, true);
    }
    else{
    // GridGenerator::subdivided_hyper_rectangle(triangulation, 0, parameters.box_size);
    //TODO for weak scaling, want to have for same depth different basin widths (and breadths)
    //get a rectangle with x_length in multiples of the z_length,(3d y too)
    // want cell structure(i.e. location of vertices to be same, just more - "grown basin")
    //GridGenerator::hyper_rectangle()
    }



    triangulation.refine_global(parameters.initial_refinement_level);
    // print_mesh_info(triangulation, "my_grid");
    for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
        cell->set_material_id(0);
    }
}

template <int dim>
void LayerMovementProblem<dim>::setup_dofs()
{
    TimerOutput::Scope t(computing_timer, "setup_dofs");


    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    //TODO: put out the sparsity patterns
    //Point<dim> direction (0,-1);
    //locally_owned_dofs = dof_handler.locally_owned_dofs();

    std::vector<types::global_dof_index> starting_indices;
    starting_indices.clear();

    std::cout<<dof_handler.n_locally_owned_dofs();


    //re-check this algorithm for AMR case
    //how to get face normals without FEFaceValues TODO
    const QMidpoint<dim - 1> face_quadrature_formula;
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_quadrature_points | update_normal_vectors |
                                     update_JxW_values);

    Tensor<1, dim> u;
    Point<dim> down;
    down(dim-1)=-1;


    std::vector< types::global_dof_index >dof_indices (fe.n_dofs_per_face(), 0);


    for (const auto &cell: dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {

            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            {
                if ((cell->face(face)->at_boundary()) || (cell->neighbor(face)->is_ghost()))
                {
                    fe_face_values.reinit(cell, face);
                    u=fe_face_values.normal_vector(0);
                    if(u*down< 0)
                    {
                        cell->face(face)->get_dof_indices(dof_indices);
                        starting_indices.insert(std::end(starting_indices),
                                                std::begin(dof_indices), std::end(dof_indices));
                    }
                }
            }
        }
    }

//  //remove duplicates by creating a set
    std::set<types::global_dof_index> no_duplicates_please (starting_indices.begin(),
            starting_indices.end());
//  //back to vector for the DoFRenumbering function
//  starting_indices.clear();
    starting_indices.assign(no_duplicates_please.begin(), no_duplicates_please.end());
//  starting_indices.insert(std::end(starting_indices),
//                                 std::begin(no_duplicates_please), std::end(no_duplicates_please));

    //starting_indices=locally_owned_dofs;
// DoFTools::extract_locally_owned_dofs(dof_handler, starting_indices);
DoFRenumbering::Cuthill_McKee(dof_handler,false, true, starting_indices);
    //Not working in parallel now
    //DoFRenumbering::downstream(dof_handler, direction, true);


    pcout << std::endl
          << "============DofHandler===============" << std::endl
          << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl
          << std::endl;

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_P()
{
    TimerOutput::Scope t(computing_timer, "setup_P");

    pcout << std::endl << "============Pressure===============" << std::endl << std::endl;

    // vector setup
    locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    temp_locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    old_temp_locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    completely_distributed_solution_P.reinit(locally_owned_dofs, mpi_communicator);

    old_locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    rhs_P.reinit(locally_owned_dofs, mpi_communicator);
    old_rhs_P.reinit(locally_owned_dofs, mpi_communicator);

    system_rhs_P.reinit(locally_owned_dofs, mpi_communicator);

    // constraints

    constraints_P.clear();

    constraints_P.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints_P);
    // zero dirichlet at top
    //the "top" has boundary_id = dim*2-1; (so 3 for 2d, 5 for 3d)
    VectorTools::interpolate_boundary_values(dof_handler, dim*2-1, ZeroFunction<dim>(),
            constraints_P); // TODO get rid of raw number
    constraints_P.close();

    // create sparsity pattern

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_P, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
            locally_relevant_dofs);
    // setup matrices

    system_matrix_P.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    laplace_matrix_P.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    mass_matrix_P.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_T()
{
    TimerOutput::Scope t(computing_timer, "setup_system_T");

    pcout << std::endl << "============Temperature===============" << std::endl << std::endl;

    // vector setup
    locally_relevant_solution_T.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    old_locally_relevant_solution_T.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    completely_distributed_solution_T.reinit(locally_owned_dofs, mpi_communicator);

    rhs_T.reinit(locally_owned_dofs, mpi_communicator);
    old_rhs_T.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs_T.reinit(locally_owned_dofs, mpi_communicator);

    // constraints

    constraints_T.clear();
    constraints_T.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints_T);
    // zero dirichlet at top
    //the "top" has boundary_id = dim*2-1; (so 3 for 2d, 5 for 3d)
    VectorTools::interpolate_boundary_values(dof_handler, dim*2-1, ZeroFunction<dim>(),
            constraints_T); // TODO again raw number for boundary_id
    // Keep top at fixed temperature, TODO check compatibility condition
    // VectorTools::interpolate_boundary_values(dof_handler_T, 3, ConstantFunction<dim>(20), constraints_T);
    constraints_T.close();

    // create sparsity pattern

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_T, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
            locally_relevant_dofs);
    // setup matrices

    system_matrix_T.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    laplace_matrix_T.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    mass_matrix_T.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_Sigma()
{
    // First of two SUPG problems
    TimerOutput::Scope t(computing_timer, "setup_system_Sigma");

    pcout << std::endl << "============Overburden===============" << std::endl << std::endl;

    // vector setup
    locally_relevant_solution_Sigma.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    old_locally_relevant_solution_Sigma.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    //temp_locally_relevant_solution_Sigma.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    rhs_Sigma.reinit(locally_owned_dofs, mpi_communicator);

    // constraints

    constraints_Sigma.clear();
    constraints_Sigma.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints_Sigma);

    // inflow bc at top
    //the "top" has boundary_id = dim*2-1; (so 3 for 2d, 5 for 3d)
//VectorTools::interpolate_boundary_values(dof_handler, 3, ConstantFunction<dim>(inflow_rate*15000),
//                                           constraints_Sigma); // TODO put in sedimentation(x,y,t)
    VectorTools::interpolate_boundary_values(dof_handler, dim*2-1, ZeroFunction<dim>(),
            constraints_Sigma);
    constraints_Sigma.close();

    // create sparsity pattern

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_Sigma, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
            locally_relevant_dofs);
    // setup matrix

    system_matrix_Sigma.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_F()
{
    // Second of two SUPG problems
    TimerOutput::Scope t(computing_timer, "setup_system_F");

    pcout << std::endl << "============Speed Function===============" << std::endl << std::endl;

    // vector setup
    locally_relevant_solution_F.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    completely_distributed_solution_F.reinit(locally_owned_dofs, mpi_communicator);

    rhs_F.reinit(locally_owned_dofs, mpi_communicator);

    // constraints

    constraints_F.clear();
    constraints_F.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints_F);
    SedimentationRate<dim> sedrate(current_time, parameters);
    sedrate.set_time(current_time);

    // inflow bc at top
    //the "top" has boundary_id = dim*2-1; (so 3 for 2d, 5 for 3d)
    VectorTools::interpolate_boundary_values(dof_handler, dim*2-1, sedrate,
            constraints_F);
    constraints_F.close();

    // create sparsity pattern

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_F, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
            locally_relevant_dofs);
    // setup matrix

    system_matrix_F.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_LS()
{
    // Note: just solution vectors here, no matrices
    TimerOutput::Scope t(computing_timer, "setup_system_LS");

    dof_handler_LS.distribute_dofs(fe_LS);

    pcout << std::endl
          << "============LEVEL SETS===============" << std::endl
          << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
          << "Number of degrees of freedom: " << dof_handler_LS.n_dofs() << std::endl
          << std::endl;

    locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_LS, locally_relevant_dofs_LS);

    // vector setup
    locally_relevant_solution_LS_0.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);

    completely_distributed_solution_LS_0.reinit(locally_owned_dofs_LS, mpi_communicator);

    // non-vertical zero vector to feed into LevelSetSolver
    locally_relevant_solution_Wxy.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);

    locally_relevant_solution_F.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
    completely_distributed_solution_F.reinit(locally_owned_dofs_LS, mpi_communicator);

    // constraints

    constraints_LS.clear();

    constraints_LS.reinit(locally_relevant_dofs_LS);

    DoFTools::make_hanging_node_constraints(dof_handler_LS, constraints_LS);

    constraints_LS.close();
}

template <int dim>
void LayerMovementProblem<dim>::initial_conditions()
{
    // Precondition: the non/ghosted vectors have been initialized, and constraints closed (in setup functions)
    //For P, T, 0 initial values

    // init condition for P (TODO should use call to VectorTools::interpolate)
    completely_distributed_solution_P = 0;
    //VectorTools::interpolate_boundary_values(dof_handler, /*top boundary*/ 3, ZeroFunction<dim>(), constraints_P);
    VectorTools::interpolate(dof_handler, ZeroFunction<dim>(), completely_distributed_solution_P);
    constraints_P.distribute(completely_distributed_solution_P);
    locally_relevant_solution_P = completely_distributed_solution_P;

    // init condition for T   //TODO
    completely_distributed_solution_T = 0;
    //VectorTools::interpolate_boundary_values(dof_handler, /*top boundary*/ 3, ZeroFunction<dim>(), constraints_T);
    VectorTools::interpolate(dof_handler, ZeroFunction<dim>(),completely_distributed_solution_T);
    constraints_T.distribute(completely_distributed_solution_T);
    locally_relevant_solution_T = completely_distributed_solution_T;

    // init condition for LS
    // all the others will share this
    completely_distributed_solution_LS_0 = 0;
    const double min_h = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(2);
    pcout <<"min_h is:"<<min_h<<std::endl;

    VectorTools::interpolate(dof_handler_LS, Initial_LS<dim>(min_h, parameters.box_size),
                             completely_distributed_solution_LS_0);
    constraints_LS.distribute(completely_distributed_solution_LS_0);
    locally_relevant_solution_LS_0 = completely_distributed_solution_LS_0;

}

template <int dim>
void LayerMovementProblem<dim>::get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
        std::vector<double>& boundary_values_LS)
{
    std::map<unsigned int, double> map_boundary_values_LS;
    unsigned int boundary_id = 0;

    // set_boundary_inlet();
    boundary_id = 10; // inlet
    // we define the inlet to be at the top, i.e. boundary_id=3
    //the "top" has boundary_id = dim*2-1; (so 3 for 2d, 5 for 3d)
    VectorTools::interpolate_boundary_values(dof_handler_LS, dim*2-1, BoundaryPhi<dim>(1.0), map_boundary_values_LS);
    boundary_values_id_LS.resize(map_boundary_values_LS.size());
    boundary_values_LS.resize(map_boundary_values_LS.size());
    std::map<unsigned int, double>::const_iterator boundary_value_LS = map_boundary_values_LS.begin();
    for (int i = 0; boundary_value_LS != map_boundary_values_LS.end(); ++boundary_value_LS, ++i) {
        boundary_values_id_LS[i] = boundary_value_LS->first;
        boundary_values_LS[i] = boundary_value_LS->second;
    }
}

template <int dim>
void LayerMovementProblem<dim>::assemble_Sigma()
{
    TimerOutput::Scope t(computing_timer, "assemble_Sigma");
    const AdvectionField<dim> advection_field;
    const QGauss<dim> quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values | update_gradients);

//    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
//                                     update_values | update_quadrature_points | update_normal_vectors |
//                                     update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> rhs_at_quad(n_q_points);
    std::vector<Tensor<1, dim>> advection_directions(n_q_points);
//    std::vector<Tensor<1, dim>> face_advection_directions(n_face_q_points);


    std::vector<double> overburden_at_quad(n_q_points);
    std::vector<double> pressure_at_quad(n_q_points);

    Point<dim> point_for_depth;
    SedimentationRate<dim> sedRate(current_time, parameters); // rate is a negative quantity

    std::vector<double> sedimentation_rate(n_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    rhs_Sigma = 0;
    system_matrix_Sigma = 0;

    int material_id;

    for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

        material_id=cell->material_id();
        fe_values.reinit(cell);
        fe_values.get_function_values(temp_locally_relevant_solution_P, pressure_at_quad);
        fe_values.get_function_values(temp_locally_relevant_solution_Sigma, overburden_at_quad);

        // TODO consider moving these properties to the quad point level, not just cell level
        const double initial_porosity = material_data.get_surface_porosity(material_id);
        const double compaction_coefficient = material_data.get_compaction_coefficient(material_id);
        const double rock_density = material_data.get_solid_density(material_id);

         sedRate.value_list(fe_values.get_quadrature_points(), sedimentation_rate, 1);
         advection_field.value_list(fe_values.get_quadrature_points(), advection_directions);

        cell_rhs = 0;
        cell_matrix = 0;
        const double delta = 1 * cell->diameter();

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            point_for_depth = fe_values.quadrature_point(q_point);
            const double hydrostatic = 9.81 * material_data.fluid_density *
                                       (parameters.box_size - point_for_depth[dim-1]);
            const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                       compaction_coefficient, hydrostatic, material_id);

            Assert(0 < hydrostatic, ExcInternalError());
            Assert(0 <= phi, ExcInternalError());
            Assert(phi < 1, ExcInternalError());

            const double rho_b = bulkdensity(phi, material_data.fluid_density, rock_density);

            rhs_at_quad[q_point] = 9.81 * rho_b;
//            if(material_id==0)
//              {
//                rhs_at_quad[q_point] =0;
//              }




            //this should point "down"
            Assert( 0 > advection_directions[q_point][dim-1], ExcInternalError());

            //Assert ( 0 <sedimentation_rate[q_point], ExcInternalError());

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)

                    cell_matrix(i, j) += ((advection_directions[q_point] * fe_values.shape_grad(j, q_point) *
                                           (fe_values.shape_value(i, q_point) +
                                            delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point)))) *
                                          fe_values.JxW(q_point));

                cell_rhs(i) += (fe_values.shape_value(i, q_point) +
                                delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point))) *
                               rhs_at_quad[q_point] * fe_values.JxW(q_point);

            }   // end i
        }     // end q

        //Rather than implement the boundary term, we specify as an essential condition on the test space
        //So it is handled in a call to the constraints_Sigma

        // For the inflow boundary term
//    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//      if (cell->face(face)->at_boundary()) {
//        fe_face_values.reinit(cell, face);

//        advection_field.value_list(fe_face_values.get_quadrature_points(), face_advection_directions);
//        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
//          // the following determines whether inflow or not
//          if (fe_face_values.normal_vector(q_point) * face_advection_directions[q_point] < 0)
//            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
//              for (unsigned int j = 0; j < dofs_per_cell; ++j)
//                cell_matrix(i, j) -= (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
//                                      fe_face_values.shape_value(i, q_point) * fe_face_values.shape_value(j, q_point) *
//                                      fe_face_values.JxW(q_point));
//              cell_rhs(i) -=
//                  (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
//                   sedimentation_rate[q_point] * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
//            }
//      }

        cell->get_dof_indices(local_dof_indices); // distribute to correct globally numbered vector

        constraints_Sigma.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_Sigma,
                rhs_Sigma);
    } // end cell loop

    rhs_Sigma.compress(VectorOperation::add);
    system_matrix_Sigma.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::solve_Sigma()
{
    TimerOutput::Scope t(computing_timer, "solve_Sigma");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-6 * rhs_Sigma.l2_norm());
    //  LA::SolverBicgstab solver(solver_control, mpi_communicator);
    LA::SolverGMRES solver(solver_control, mpi_communicator);
    //  LA::MPI::PreconditionAMG preconditioner;
    //  LA::MPI::PreconditionAMG::AdditionalData data;
    //  LA::MPI::PreconditionSSOR preconditioner;
    //  LA::MPI::PreconditionSSOR::AdditionalData data;
    //LA::MPI::PreconditionJacobi preconditioner;
    //LA::MPI::PreconditionJacobi::AdditionalData data;
    //LA::PreconditionSSOR preconditioner;
    //does not compile with this
    //LA::MPI::PreconditionBlockJacobi preconditioner;
    //LA::PreconditionBlockJacobi preconditioner;
    //does with this
    PETScWrappers::PreconditionBlockJacobi preconditioner;
    PETScWrappers::PreconditionBlockJacobi::AdditionalData data;
    //data.symmetric_operator = false;
    preconditioner.initialize(system_matrix_Sigma, data);

    solver.solve(system_matrix_Sigma, completely_distributed_solution, rhs_Sigma, preconditioner);

    pcout << " Overburden supg system solved in " << solver_control.last_step() << " iterations." << std::endl;

    constraints_Sigma.distribute(completely_distributed_solution);
    //old_locally_relevant_solution_Sigma=locally_relevant_solution_Sigma;
    temp_locally_relevant_solution_Sigma = completely_distributed_solution;


}

template <int dim>
void LayerMovementProblem<dim>::assemble_F()
{

    TimerOutput::Scope t(computing_timer, "assemble_F");
    const AdvectionField<dim> advection_field;
    const QGauss<dim> quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values | update_gradients);

//    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
//                                     update_values | update_quadrature_points | update_normal_vectors |
//                                     update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> rhs_at_quad(n_q_points);
    std::vector<Tensor<1, dim>> advection_directions(n_q_points);
//    std::vector<Tensor<1, dim>> face_advection_directions(n_face_q_points);


    std::vector<double> overburden_at_quad(n_q_points);
    std::vector<double> old_overburden_at_quad(n_q_points);
    std::vector<double> pressure_at_quad(n_q_points);
    std::vector<double> old_pressure_at_quad(n_q_points);

    Point<dim> point_for_depth;
    SedimentationRate<dim> sedRate(current_time, parameters);

    std::vector<double> sedimentation_rate(n_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    rhs_F = 0;
    system_matrix_F = 0;

    int material_id;

    for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

        material_id=cell->material_id();

        fe_values.reinit(cell);

        fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
        fe_values.get_function_values(locally_relevant_solution_Sigma, overburden_at_quad);
        fe_values.get_function_values(old_locally_relevant_solution_P, old_pressure_at_quad);
        fe_values.get_function_values(old_locally_relevant_solution_Sigma, old_overburden_at_quad);

        // TODO consider moving these properties to the quad point level, not just cell level
        const double initial_porosity = material_data.get_surface_porosity(material_id);
        const double compaction_coefficient = material_data.get_compaction_coefficient(material_id);
        advection_field.value_list(fe_values.get_quadrature_points(), advection_directions);
        sedRate.value_list(fe_values.get_quadrature_points(), sedimentation_rate, 1);


        cell_rhs = 0;
        cell_matrix = 0;
        const double delta = 1 * cell->diameter();

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            point_for_depth = fe_values.quadrature_point(q_point);
            const double hydrostatic = 9.81 * material_data.fluid_density *
                                       (parameters.box_size - point_for_depth[dim-1]);
            Assert(0 < hydrostatic, ExcInternalError());
            const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                        compaction_coefficient, hydrostatic, material_id);

            Assert(0 <= phi, ExcInternalError());
            Assert(phi < 1, ExcInternalError());

            const double old_phi = porosity(old_pressure_at_quad[q_point], old_overburden_at_quad[q_point], initial_porosity,
                                            compaction_coefficient, hydrostatic, material_id);

            Assert(0 <= old_phi, ExcInternalError());
            Assert(old_phi < 1, ExcInternalError());

             double dphidt = (phi - old_phi) / time_step;



            Assert(dphidt < 0.1, ExcInternalError());

            rhs_at_quad[q_point] = -1*dphidt / (1 - phi);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)

                    cell_matrix(i, j) += ((advection_directions[q_point] * fe_values.shape_grad(j, q_point) *
                                           (fe_values.shape_value(i, q_point) +
                                            delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point)))) *
                                          fe_values.JxW(q_point));

                cell_rhs(i) += (fe_values.shape_value(i, q_point) +
                                delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point))) *
                               rhs_at_quad[q_point] * fe_values.JxW(q_point);

            }   // end i
        }     // end q

        // For the inflow boundary term

//    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//      if (cell->face(face)->at_boundary()) {
//        fe_face_values.reinit(cell, face);

//        advection_field.value_list(fe_face_values.get_quadrature_points(), face_advection_directions);
//        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
//          // the following determines whether inflow or not
//          if (fe_face_values.normal_vector(q_point) * face_advection_directions[q_point] < 0)
//            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
//              for (unsigned int j = 0; j < dofs_per_cell; ++j)
//                cell_matrix(i, j) -= (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
//                                      fe_face_values.shape_value(i, q_point) * fe_face_values.shape_value(j, q_point) *
//                                      fe_face_values.JxW(q_point));
//              cell_rhs(i) -=
//                  (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
//                   sedimentation_rate[q_point] * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
//            }
//      }

        cell->get_dof_indices(local_dof_indices); // distribute to correct globally numbered vector

        constraints_F.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_F, rhs_F);
    } // end cell loop

    rhs_F.compress(VectorOperation::add);
    system_matrix_F.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::solve_F()
{
    TimerOutput::Scope t(computing_timer, "solve_F");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-6 * rhs_F.l2_norm());
    //  LA::SolverBicgstab solver(solver_control, mpi_communicator);
    LA::SolverGMRES solver(solver_control, mpi_communicator);
    //  LA::MPI::PreconditionAMG preconditioner;
    //  LA::MPI::PreconditionAMG::AdditionalData data;
    //  LA::MPI::PreconditionSSOR preconditioner;
    //  LA::MPI::PreconditionSSOR::AdditionalData data;
    PETScWrappers::PreconditionBlockJacobi preconditioner;
    PETScWrappers::PreconditionBlockJacobi::AdditionalData data;
//  LA::MPI::PreconditionJacobi preconditioner;
//  LA::MPI::PreconditionJacobi::AdditionalData data;

    // data.symmetric_operator = false;
    preconditioner.initialize(system_matrix_F, data);

    solver.solve(system_matrix_F, completely_distributed_solution, rhs_F, preconditioner);

    pcout << " Speed function system solved in " << solver_control.last_step() << " iterations." << std::endl;

    constraints_F.distribute(completely_distributed_solution);

    locally_relevant_solution_F = completely_distributed_solution;
}


// TODO fold this into P,F,Sigma, T assemblies to assign at quad point level
template <int dim>
void LayerMovementProblem<dim>::setup_material_configuration()
{
    //TODO
    TimerOutput::Scope t(computing_timer, "set_material_configuration");
    // This function sets material ids of cells based on the location of the interface, i.e. loc_rel_solution_LS

    const QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(fe_LS, quadrature_formula, update_values | update_quadrature_points);

    //  const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    //  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    //TODO: a better structure than vec(vec))
    std::vector<std::vector<double>> LS_at_quad (n_layers, std::vector<double>(n_q_points));

    // std::vector<double> bulkdensity_at_quad(n_q_points);
    // double eps= GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2);
    //      const double eps=0.001;
    //        double H=0;
    //          // get rho, nu
    //          if (phi>eps)
    //            H=1;
    //          else if (phi<-eps)
    //            H=-1;
    //          else
    //            H=phi/eps;
    //          diff_coeff=1000*(1+H)/2.+10*(1-H)/2.;

    // std::vector<double> id_sum(5);
    std::vector<double> id_sum(n_layers, 0);

    for (auto cell : filter_iterators(dof_handler_LS.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

        std::fill(id_sum.begin(), id_sum.end(), 0);

        fe_values.reinit(cell);
        int i=0;//for n_layers
        for(auto & layer_sol : layers_solutions)
        {
            fe_values.get_function_values( *layer_sol, LS_at_quad[i]);

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {

                Assert(LS_at_quad[i][q_point] < 1.5, ExcInternalError());
                Assert(-1.5 < LS_at_quad[i][q_point], ExcInternalError());
                //do the y=2x-1 switch so the -/+ still works
                id_sum[i] += 2*LS_at_quad[i][q_point]-1;
            }
            ++i;
            //TODO representation of interface (0 level set or 0.5, etc.) needs to be taken
            //into account in this averaging, as above for 0.5
        }
        //defining the negative to be below an interface,
        //if a LS has takes a positive value on the cell
        //it is added to the counter (cell is "inside" the layer)
        //the innermost layer is the material id
        int counter{0};
        for (i=0; i<n_layers; ++i)
        {

            if(id_sum[i]>0)
            {
                ++counter;
            }
        }
        cell->set_material_id(counter);

    } // end cell loop
}

template <int dim>
void LayerMovementProblem<dim>::assemble_matrices_P()
{
    TimerOutput::Scope t(computing_timer, "assembly_P");
    const QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Point<dim> point_for_depth;

    SedimentationRate<dim> sedRate(current_time, parameters );

    std::vector<double> pressure_at_quad(n_q_points);
    std::vector<double> overburden_at_quad(n_q_points);
    std::vector<double> old_overburden_at_quad(n_q_points);
    std::vector<double> old_pressure_at_quad(n_q_points);
    std::vector<double> sedimentation_rates(n_q_points);

    int material_id;

    for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
        cell_laplace_matrix = 0;
        cell_mass_matrix = 0;
        cell_rhs = 0;

        material_id=cell->material_id();

        fe_values.reinit(cell);

        fe_values.get_function_values(temp_locally_relevant_solution_P, pressure_at_quad);
        fe_values.get_function_values(temp_locally_relevant_solution_Sigma, overburden_at_quad);
        fe_values.get_function_values(old_locally_relevant_solution_P, old_pressure_at_quad);
        fe_values.get_function_values(old_locally_relevant_solution_Sigma, old_overburden_at_quad);

        // TODO consider moving these properties to the quad point level, not just cell level
        const double initial_porosity = material_data.get_surface_porosity(material_id);
        const double compaction_coefficient = material_data.get_compaction_coefficient(material_id);
        const double initial_permeability = material_data.get_surface_permeability(material_id);
        sedRate.value_list(fe_values.get_quadrature_points(), sedimentation_rates, 1);


       // const double compress = compressibility()   material_data.get_compressibility_coefficient(material_id);


        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
//        if(0 <= pressure_at_quad[q_point]){
//            output_vectors();
//            abort();
//          }
           // Assert( -0.1 <pressure_at_quad[q_point], ExcInternalError());
            Assert(-0.1 <overburden_at_quad[q_point], ExcInternalError());

            point_for_depth = fe_values.quadrature_point(q_point);
            const double hydrostatic = 9.81 * material_data.fluid_density *
                                       (parameters.box_size - point_for_depth[dim-1]);
            Assert(0 < hydrostatic, ExcInternalError());
            const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                        compaction_coefficient, hydrostatic, material_id);

           const double compressibilit = compressibility(phi, compaction_coefficient, material_id);


            Assert(0 <= phi, ExcInternalError());
            Assert(phi < 1, ExcInternalError());
            //Assert(0< (overburden_at_quad[q_point]-pressure_at_quad[q_point]-hydrostatic), ExcInternalError());

            const double perm_k = permeability(phi, initial_permeability, initial_porosity, material_id);
            // pcout<<"perm_k"<<perm_k<<"init<<"<< initial_permeability<<std::endl;
            Assert(0 <= perm_k, ExcInternalError());
            // Assert(perm_k <= initial_permeability, ExcInternalError());

//            const double old_phi = porosity(old_pressure_at_quad[q_point], old_overburden_at_quad[q_point], initial_porosity,
//                                            compaction_coefficient, hydrostatic, material_id);
//            Assert(0 <= old_phi, ExcInternalError());
//            Assert(old_phi < 1, ExcInternalError());

//            const double dphidt = (phi - old_phi) / time_step;

            //Assert(dphidt <= 0, ExcInternalError());

            const double diff_coeff_at_quad = (perm_k / (material_data.fluid_viscosity * compaction_coefficient *(1-phi)) );
             double rhs_coeff = 1;
            const double rhs_at_quad = (9.8 * (2220- material_data.fluid_density)* -1*sedimentation_rates[q_point]);
              // (overburden_at_quad[q_point] - old_overburden_at_quad[q_point]) / ( time_step) -
              //                       (9.8 * material_data.fluid_density * -1*sedimentation_rates[q_point]);
//            if(material_id==0)
//              {
//                rhs_coeff=0;
//              }
            //Assert (0 <= rhs_at_quad, ExcInternalError());

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                    cell_laplace_matrix(i, j) += diff_coeff_at_quad * (fe_values.shape_grad(i, q_point) *
                                                 fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

                    cell_mass_matrix(i, j) +=
                        (fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));
                } //end of j

                cell_rhs(i) += rhs_coeff*(rhs_at_quad * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
            } //end of i
        } // end q

        cell->get_dof_indices(local_dof_indices);
        constraints_P.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_P, rhs_P);
        constraints_P.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_P);
    } // end cell

    laplace_matrix_P.compress(VectorOperation::add);
    mass_matrix_P.compress(VectorOperation::add);
    rhs_P.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::forge_system_P()
{
    // in this function we manipulate A, M, F, resulting from assemble_matrices_P
    TimerOutput::Scope t(computing_timer, "forge_P");
    LA::MPI::Vector tmp;
    LA::MPI::Vector forcing_terms;

    tmp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    forcing_terms.reinit(locally_owned_dofs, mpi_communicator);

    //the statement below is now placed into the prepare_next_time_step method
    //old_locally_relevant_solution_P = locally_relevant_solution_P;

    mass_matrix_P.vmult(system_rhs_P, old_locally_relevant_solution_P);

    laplace_matrix_P.vmult(tmp, old_locally_relevant_solution_P);
    //  pcout << "laplace symmetric: " << laplace_matrix.is_symmetric()<<std::endl;
    system_rhs_P.add(-(1 - theta) * time_step*2, tmp);

    forcing_terms.add(2*time_step * theta, rhs_P);

    forcing_terms.add(2* time_step * (1 - theta), old_rhs_P);

    system_rhs_P += forcing_terms;
    system_rhs_P.compress (VectorOperation::add);

    system_matrix_P.copy_from(mass_matrix_P);
    // system_matrix.compress (VectorOperation::add);

    system_matrix_P.add(2*time_step * theta, laplace_matrix_P);
    system_matrix_P.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::solve_time_step_P()
{
    TimerOutput::Scope t(computing_timer, "solve_time_step_P");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * system_rhs_P.l2_norm());
    LA::SolverCG solver(solver_control, mpi_communicator);

    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

    data.symmetric_operator = true;
    preconditioner.initialize(system_matrix_P, data);

    solver.solve(system_matrix_P, completely_distributed_solution, system_rhs_P, preconditioner);

    pcout << " Pressure system solved in " << solver_control.last_step() << " iterations." << std::endl;

    constraints_P.distribute(completely_distributed_solution);

    temp_locally_relevant_solution_P = completely_distributed_solution;
}

template <int dim>
void LayerMovementProblem<dim>::assemble_matrices_T()
{
    TimerOutput::Scope t(computing_timer, "assembly_T");
    const QGauss<dim> quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_quadrature_points | update_normal_vectors |
                                     update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Point<dim> point_for_depth;

    std::vector<double> pressure_at_quad(n_q_points);
    std::vector<double> overburden_at_quad(n_q_points);
    std::vector<double> bulkheat_capacity_at_quad(n_q_points);
    std::vector<double> thermal_conductivity_at_quad(n_q_points);

    int material_id;

    for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
        cell_laplace_matrix = 0;
        cell_mass_matrix = 0;
        cell_rhs = 0;

        material_id = cell->material_id();

        fe_values.reinit(cell);
        fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
        fe_values.get_function_values(locally_relevant_solution_Sigma, overburden_at_quad);

        // TODO consider moving these properties to the quad point level, not just cell level
        const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
        const double compaction_coefficient = material_data.get_compaction_coefficient(cell->material_id());
        const double rock_density = material_data.get_solid_density(cell->material_id());
        const double heat_capacity = material_data.get_heat_capacity(cell->material_id());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            point_for_depth = fe_values.quadrature_point(q_point);
            const double hydrostatic = 9.81 * material_data.fluid_density *
                                       (parameters.box_size - point_for_depth[dim-1]); // TODO make this dim independent
            const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                        compaction_coefficient, hydrostatic, material_id);
            //Assert(0 < phi < 1, ExcInternalError());

            const double rho_b = bulkdensity(phi, material_data.fluid_density, rock_density);
            const double bulk_hc = bulkheatcapacity(phi, material_data.fluid_heat_capacity, heat_capacity);

            //      fe_values.get_function_values(thermal_conductivity, thermal_conductivity_at_quad);
            // TODO
            const double diff_coeff_at_quad = 10;
            //  thermal_conductivity_at_quad[q_point] /(bulkheat_capacity_at_quad[q_point]*bulkdensity_at_quad[q_point] );
            const double rhs_at_quad = 0; // TODO bottom boundary flux from parameter file

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                    cell_laplace_matrix(i, j) += diff_coeff_at_quad * fe_values.shape_grad(i, q_point) *
                                                 fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);

                    cell_mass_matrix(i, j) += (rho_b * bulk_hc * fe_values.shape_value(i, q_point) *
                                               fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));
                } // end j

                //          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_point)) *
                //                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
                cell_rhs(i) += (rhs_at_quad * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));

            }   // end i
        }     // end q

        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number) {
            if (cell->face(face_number)->at_boundary() &&
                    (cell->face(face_number)->boundary_id() == 2)) // bottom of domain TODO remove raw number
            {
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
                    //                const double neumann_value
                    //                  = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                    //                     fe_face_values.normal_vector(q_point));
                    // TODO pick the right flux value
                    const double neumann_value = 100; // represents the bottom flux boundary condition
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        cell_rhs(i) += (neumann_value * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
                    }
                }
            }
        } // end face loop

        cell->get_dof_indices(local_dof_indices);
        constraints_T.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_T, rhs_T);
        constraints_T.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_T);
    } // end cell

    laplace_matrix_T.compress(VectorOperation::add);
    mass_matrix_T.compress(VectorOperation::add);
    rhs_T.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::forge_system_T()
{
    // in this function we manipulate A, M, F, resulting from assemble_matrices_T
    TimerOutput::Scope t(computing_timer, "forge_T");
    LA::MPI::Vector tmp;
    LA::MPI::Vector forcing_terms;

    tmp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    forcing_terms.reinit(locally_owned_dofs, mpi_communicator);

    old_locally_relevant_solution_T = locally_relevant_solution_T;
    mass_matrix_T.vmult(system_rhs_T, old_locally_relevant_solution_T);

    laplace_matrix_T.vmult(tmp, old_locally_relevant_solution_T);
    //  pcout << "laplace symmetric: " << laplace_matrix.is_symmetric()<<std::endl;
    system_rhs_T.add(-(1 - theta) * time_step, tmp);

    forcing_terms.add(time_step * theta, rhs_T);

    forcing_terms.add(time_step * (1 - theta), old_rhs_T);

    system_rhs_T += forcing_terms;
    // system_matrix.compress (VectorOperation::add);

    system_matrix_T.copy_from(mass_matrix_T);
    // system_matrix.compress (VectorOperation::add);

    system_matrix_T.add(laplace_matrix_T, time_step * (1 - theta));
}

template <int dim>
void LayerMovementProblem<dim>::solve_time_step_T()
{
    TimerOutput::Scope t(computing_timer, "solve_time_step_T");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * system_rhs_T.l2_norm());
    LA::SolverCG solver(solver_control, mpi_communicator);

    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

    data.symmetric_operator = true;
    preconditioner.initialize(system_matrix_T, data);

    solver.solve(system_matrix_T, completely_distributed_solution, system_rhs_T, preconditioner);

    pcout << " Temperature system solved in " << solver_control.last_step() << " iterations." << std::endl;

    constraints_T.distribute(completely_distributed_solution);

    locally_relevant_solution_T = completely_distributed_solution;
}


template <int dim>
bool LayerMovementProblem<dim>::estimate_nl_error()
{
  const double tolerance = 10e-4; //parameters.
  double residual = 1;

}



template <int dim>
void LayerMovementProblem<dim>::prepare_next_time_step()
{
    old_locally_relevant_solution_P=locally_relevant_solution_P;
    old_locally_relevant_solution_Sigma=locally_relevant_solution_Sigma;
    old_locally_relevant_solution_T=locally_relevant_solution_T;
    old_rhs_P=rhs_P;
}

template <int dim>
void LayerMovementProblem<dim>::output_vectors_LS()
{
    TimerOutput::Scope t(computing_timer, "output_LS");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_LS);
    int i=0;
    for( auto & layer_sol : layers_solutions)
    {
        std::string layer_out = "LS"+  Utilities::int_to_string(i, 3);
        data_out.add_data_vector(*layer_sol, layer_out);
        ++i;
    }
//    LA::MPI::Vector ng_material_kind;
//    ng_material_kind.reinit(locally_owned_dofs,  mpi_communicator);
//    LA::MPI::Vector g_material_kind;
//    g_material_kind.reinit(locally_owned_dofs,locally_relevant_dofs,  mpi_communicator);

//    //std::vector<unsigned int> material_kind(triangulation.n_active_cells());
//     i = 0;
//    for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
//    ng_material_kind[i]=cell->material_id();
//      ++i;
//    }
//    ng_material_kind.compress(VectorOperation::insert);
//    g_material_kind=ng_material_kind;
//    ComputePorosity<dim> porosity;



    data_out.add_data_vector(locally_relevant_solution_LS_0, "LS");
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    const std::string filename = ("sol_LS_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                  Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
            filenames.push_back("sol_LS_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(i, 4) + ".vtu");

        std::ofstream master_output(("sol_LS_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
    }
}



template <int dim>
void LayerMovementProblem<dim>::output_vectors()
{
    TimerOutput::Scope t(computing_timer, "output");
    // output_number++;
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution_P, "P");
    data_out.add_data_vector(old_locally_relevant_solution_P, "old_P");
    data_out.add_data_vector(locally_relevant_solution_T, "T");
    data_out.add_data_vector(old_locally_relevant_solution_T, "old_T");
    data_out.add_data_vector(locally_relevant_solution_Sigma, "Sigma");
    data_out.add_data_vector(old_locally_relevant_solution_Sigma, "old_Sigma");
    data_out.add_data_vector(locally_relevant_solution_F, "F");

    //abuse the temp_locally_relevant_Sigma ghosted vector to output non-ghosted rhs_Sigma
    temp_locally_relevant_solution_Sigma=rhs_Sigma;
    data_out.add_data_vector(old_locally_relevant_solution_Sigma, "rhsSigma");

//  data_out.add_data_vector(system_rhs_P, "s_rhs_P");
//  data_out.add_data_vector(rhs_F, "rhs_F" );
//  data_out.add_data_vector(rhs_Sigma, "rhs_s");



//      Vector<float> material_id(triangulation.n_active_cells());
//      for (unsigned int i = 0; i < material_id.size(); ++i)
//          material_id(i) = cell->material_id();
//      data_out.add_data_vector(material_id, "material_id");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    const std::string filename = ("sol_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                  Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
            filenames.push_back("sol_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(i, 4) + ".vtu");

        std::ofstream master_output(("sol_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
    }
}

template <int dim>
int LayerMovementProblem<dim>::active_layers_in_time (double time)
{
  //equitemporal division over layers
  for (int i=1;i<=n_layers;++i)
    {
      double current_fraction= static_cast<double>(i)/(n_layers);
      if(time<(current_fraction*final_time/2))
        {
          pcout<<"layer"<<i<<std::endl;
          return i;
        }

    }

}




template <int dim>
class LayerMovementProblem<dim>::Postprocessor : public DataPostprocessor<dim>
{
public:
  Postprocessor (const CPPLS::MaterialData& material_data,
                 const CPPLS::Parameters& parameters);
  virtual
  void
  evaluate_vector_field
  (const DataPostprocessorInputs::Vector<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const override;
  virtual std::vector<std::string> get_names () const override;
  virtual
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation () const override;
  virtual UpdateFlags get_needed_update_flags () const override;
private:
  const CPPLS::MaterialData& material_data;
  const CPPLS::Parameters& parameters;
};
template <int dim>
LayerMovementProblem<dim>::Postprocessor::
Postprocessor (const CPPLS::MaterialData& material_data,
               const CPPLS::Parameters& parameters)
  :
  material_data (material_data),
  parameters (parameters)
{}
template <int dim>
std::vector<std::string>
LayerMovementProblem<dim>::Postprocessor::get_names() const
{
  std::vector<std::string> solution_names;

  solution_names.push_back ("porosity");
  solution_names.push_back ("permeability");
  solution_names.push_back ("VES");
  solution_names.push_back ("material");
  solution_names.push_back ("hydrostatic");
  solution_names.push_back ("overpressure");
  solution_names.push_back ("overburden");
  solution_names.push_back ("pore_pressure");
  solution_names.push_back ("speed_function");



  //solution_names.push_back ("T");

  return solution_names;
}
template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
LayerMovementProblem<dim>::Postprocessor::
get_data_component_interpretation () const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation;
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);

  return interpretation;
}
template <int dim>
UpdateFlags
LayerMovementProblem<dim>::Postprocessor::get_needed_update_flags() const
{
  return update_values | update_gradients | update_q_points;
}
template <int dim>
void
LayerMovementProblem<dim>::Postprocessor::
evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert (inputs.solution_gradients.size() == n_quadrature_points,
          ExcInternalError());
  Assert (computed_quantities.size() == n_quadrature_points,
          ExcInternalError());
  Assert (inputs.solution_values[0].size() == 3,
          ExcInternalError());

    //cell properties
  const typename DoFHandler<dim>::cell_iterator
    current_cell = inputs.template get_cell<DoFHandler<dim>>();
  const unsigned int mat_id = current_cell->material_id();
//  const Point<dim> center = current_cell->center();
//  const double depth = parameters.box_size- center[1];

  const double initial_porosity = material_data.get_surface_porosity(mat_id);
  const double compaction_coefficient = material_data.get_compaction_coefficient(mat_id);
  const double initial_permeability = material_data.get_surface_permeability(mat_id);

  for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      //point values
      const Point<dim> point_for_depth = inputs.evaluation_points[q];
      const double hydrostatic = 9.81*material_data.fluid_density*(parameters.box_size - point_for_depth[dim-1]);//point_for_depth;

      //relabel the incoming components
      const double overpressure=inputs.solution_values[q](0);
      const double sigma = inputs.solution_values[q](1);
      const double speed_function = inputs.solution_values[q](2);


    //TODO: using the std::function in the LayerMovementProblem class is not working within this inherited DataPostProcessor class
    // so as a work around we branch based on input file
//      //porosity
//      computed_quantities[q](0)
//          = LayerMovementProblem<dim>::porosity(overpressure, sigma, initial_porosity, compaction_coefficient, hydrostatic);

      //porosity
      if
      (parameters.linear_in_void_ratio)
      {
         // computed_quantities[q](0)=


      }
      //Athy's law
      else
      {
          computed_quantities[q](0)=initial_porosity *
              (std::exp(-1 * compaction_coefficient * (sigma - overpressure - hydrostatic)));
      }



      //permeability
      computed_quantities[q](1)
          = CPPLS::permeability(computed_quantities[q](0), initial_permeability, initial_porosity, mat_id);
      //VES
      computed_quantities[q](2)
          = CPPLS::VES(sigma, overpressure, hydrostatic, mat_id);
      //material_id
      computed_quantities[q](3)
          =mat_id;
      computed_quantities[q](4)
          =hydrostatic;
      computed_quantities[q](5)
          =overpressure;
      computed_quantities[q](6)
          =sigma;
      computed_quantities[q](7)
          =overpressure+hydrostatic;
      computed_quantities[q](8)
          =speed_function;



    }
}
template <int dim>
void LayerMovementProblem<dim>::output_results_pp ()
{
  TimerOutput::Scope t(computing_timer, "output_pp");
  //computing_timer.enter_section ("Postprocessing");
  //the purpose of this is to create a vector-valued solution vector, composed of
  //gluing together the scalar solution vectors(i.e., pressure, overburden and speed function)
  //for use in the Postprocessor class.
  //Note these all share one DoFHandler (i.e., dof_handler), so there might be a better way to make the
  //vector-valued solution.
  //The current method would allow for joining in the dof_handler_LS solution vectors


  const FESystem<dim> joint_fe(fe, 1, fe, 1, fe, 1);
  //FESystem<dim, dim> joint_fe (FE_Q<dim>(2), 2);

  DoFHandler<dim> joint_dof_handler (triangulation);
  joint_dof_handler.distribute_dofs (joint_fe);
  Assert (joint_dof_handler.n_dofs() ==
          dof_handler.n_dofs()*3,
          ExcInternalError());
  LA::MPI::Vector joint_solution;
  joint_solution.reinit (joint_dof_handler.locally_owned_dofs(), mpi_communicator);
  {
    std::vector<types::global_dof_index> local_joint_dof_indices (joint_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);

    //std::vector<types::global_dof_index> local_temperature_dof_indices (temperature_fe.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    joint_cell       = joint_dof_handler.begin_active(),
    joint_endc       = joint_dof_handler.end(),
    cell      = dof_handler.begin_active();
//    temperature_cell = temperature_dof_handler.begin_active();
    for (; joint_cell!=joint_endc;
         ++joint_cell, ++cell/*, ++temperature_cell*/)
      if (joint_cell->is_locally_owned())
        {
          joint_cell->get_dof_indices (local_joint_dof_indices);
          cell->get_dof_indices (local_dof_indices);
//          temperature_cell->get_dof_indices (local_temperature_dof_indices);
          for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
            {
            if (joint_fe.system_to_base_index(i).first.first == 0)
              {
                Assert (joint_fe.system_to_base_index(i).second
                        <
                        local_dof_indices.size(),
                        ExcInternalError());
                joint_solution(local_joint_dof_indices[i])
                  = locally_relevant_solution_P(local_dof_indices
                                    [joint_fe.system_to_base_index(i).second]);
              }
            else if (joint_fe.system_to_base_index(i).first.first == 1)
              {

                Assert (joint_fe.system_to_base_index(i).second
                        <
                        local_dof_indices.size(),
                        ExcInternalError());
                joint_solution(local_joint_dof_indices[i])
                  = locally_relevant_solution_Sigma(local_dof_indices
                                         [joint_fe.system_to_base_index(i).second]);
              }
              else
              {
                Assert (joint_fe.system_to_base_index(i).first.first == 2,
                        ExcInternalError());
                Assert (joint_fe.system_to_base_index(i).second
                        <
                        local_dof_indices.size(),
                        ExcInternalError());
                joint_solution(local_joint_dof_indices[i])
                  = locally_relevant_solution_F(local_dof_indices
                                         [joint_fe.system_to_base_index(i).second]);


               }

            }
        }
  }

  joint_solution.compress(VectorOperation::insert);
  IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
  DoFTools::extract_locally_relevant_dofs (joint_dof_handler, locally_relevant_joint_dofs);
  LA::MPI::Vector locally_relevant_joint_solution;

  locally_relevant_joint_solution.reinit (joint_dof_handler.locally_owned_dofs(), locally_relevant_joint_dofs, mpi_communicator);
  locally_relevant_joint_solution = joint_solution;
  Postprocessor postprocessor ( material_data, parameters);
  DataOut<dim> data_out;
  data_out.attach_dof_handler (joint_dof_handler);
  data_out.add_data_vector (locally_relevant_joint_solution, postprocessor);
  data_out.build_patches ();
  static int out_index=0;
  const std::string filename = ("solution-" +
                                Utilities::int_to_string (out_index, 5) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4) +
                                ".vtu");
  std::ofstream output (filename.c_str());
  data_out.write_vtu (output);
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        filenames.push_back (std::string("solution-") +
                             Utilities::int_to_string (out_index, 5) +
                             "." +
                             Utilities::int_to_string(i, 4) +
                             ".vtu");
      const std::string
      pvtu_master_filename = ("solution-" +
                              Utilities::int_to_string (out_index, 5) +
                              ".pvtu");
      std::ofstream pvtu_master (pvtu_master_filename.c_str());
      data_out.write_pvtu_record (pvtu_master, filenames);
//      const std::string
//      visit_master_filename = ("solution-" +
//                               Utilities::int_to_string (out_index, 5) +
//                               ".visit");
//      std::ofstream visit_master (visit_master_filename.c_str());
//      DataOutBase::write_visit_record (visit_master, filenames);
    }
  //computing_timer.exit_section ();
 out_index++;
}




template <int dim>
void LayerMovementProblem<dim>::run()
{
  constexpr double seconds_in_Myear{60*60*24*365.25*1e6};
  pcout<<"CPPLS running in "<<dim<<" dimensions"<<std::endl;

    //this sets porosity, compressibility, permeability based on choices in parameter file
    set_physical_functions();

    // common mesh
    setup_geometry();
    // common dofhandler, except for LS
    setup_dofs();
    // these are the 4 systems treated in this code
    setup_system_P();
    setup_system_T();
    setup_system_Sigma();
    setup_system_F();
    // the solution of this system is done in the LevelSetSolver class
    setup_system_LS();

    initial_conditions();
    const unsigned int output_interval= parameters.output_interval;
    const bool compute_temperature = parameters.compute_temperature;

    const unsigned int maxiter = parameters.maxiter;
    const double nl_tol = parameters.nl_tol;


    //const SedimentationRate SedRate(parameters);
    const double base_sedimentation_rate = parameters.base_sedimentation_rate;
    // initialize level set solver
    // we use some hardcode defaults for now

    const double min_h = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(2);
    const double cfl = parameters.cfl;
    const double umax = base_sedimentation_rate;  //max_sedRate
    time_step = cfl * min_h / umax;
    // pcout<<"min_h"<<min_h;


    const double cK = 1.0;//compression coeff
    const double cE = 1.0;//entropy-visc coeff (non-dimensional cf. p 452 (around eq 18) Guermond, 2017)
    const bool verbose = true;
    std::string ALGORITHM = "MPP_uH";
    const unsigned int TIME_INTEGRATION = 1; // corresponds to SSP33


    n_layers=parameters.n_layers;
    int n_active_layers=0;


    // BOUNDARY CONDITIONS FOR LS
    get_boundary_values_LS(boundary_values_id_LS, boundary_values_LS);

    locally_relevant_solution_F = -1*base_sedimentation_rate;


    //assume locally_relevant_solution_LS_0 is a good initial value for all level sets

    for(int i=0; i<n_layers; ++i)
    {
        layers.emplace_back(new LevelSetSolver<dim>(degree_LS, degree, time_step,
                            cK, cE, verbose, ALGORITHM, TIME_INTEGRATION,
                            triangulation, mpi_communicator,
                            dof_handler, dof_handler_LS, computing_timer, i));
        layers_solutions.emplace_back(new LA::MPI::Vector);
        layers_solutions[i]->reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);

        layers[i]->set_boundary_conditions(boundary_values_id_LS, boundary_values_LS);
        if(dim==3){
        layers[i]->initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Wxy,
                                     locally_relevant_solution_Wxy, locally_relevant_solution_F);
          }
        else
          {
            layers[i]->initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Wxy,
                                        locally_relevant_solution_F);

          }
    }

    //display_vectors();

    // TIME STEPPING
    timestep_number = 1;
    for (double time = time_step; time <= final_time; time += time_step, ++timestep_number) {
        pcout << "Time step " << timestep_number << " at t=" << time <<"s: "<<time/seconds_in_Myear<<"Ma"<< std::endl;
        pcout<< " % complete:"<<100*(timestep_number*time_step)/final_time<<std::endl; //for constant time_step
        Assert (time_step< final_time, ExcNotImplemented());

        current_time=time;

        // Solve for F the scalar speed function which is passed to the LevelSetSolver
        // which expects the vector (wx,wy) or (wx,wy,wz)
        // dim=2 we pass F as wy and dim=3 we pass F as wz with 0 otherwise

        // Level set computation
        // original level_set_solver.set_velocity(locally_relevant_solution_u, locally_relevant_solution_v);
        for(int h=0;h<2;++h)
          {
        n_active_layers=active_layers_in_time(time);
        {
            //TimerOutput::Scope t(computing_timer, "LS");
            for(int i=0; i<n_active_layers; ++i)
            {

                if(dim==3){
                layers[i]->set_velocity(locally_relevant_solution_Wxy,locally_relevant_solution_Wxy, locally_relevant_solution_F);
                  }
                else{
                  layers[i]->set_velocity(locally_relevant_solution_Wxy, locally_relevant_solution_F);
                  }

                layers[i]->nth_time_step();
                layers[i]->get_unp1(locally_relevant_solution_LS_0);
                (*layers_solutions[i])=locally_relevant_solution_LS_0;

            }
        }
          }

        // set material ids based on locally_relevant_solution_LS
        setup_material_configuration(); // TODO: move away from cell id to values at quad points

        //output_results_pp();

        //prepare for nonlinear Picard iteration
        temp_locally_relevant_solution_Sigma=locally_relevant_solution_Sigma;
        temp_locally_relevant_solution_P=locally_relevant_solution_P;
        bool is_converged=false;
        int nl_loop_count=0;
        int maxiter{20};
        double tolerance =1e-2;
        double deviation{0};


        while (is_converged==false && nl_loop_count<maxiter)
        {
            old_temp_locally_relevant_solution_P=temp_locally_relevant_solution_P;
            assemble_Sigma();
            solve_Sigma();    // generates temp_l_r_s_Sigma

            // pressure solution
            assemble_matrices_P();
            forge_system_P();
            solve_time_step_P(); // temp_loc_r_s_P

            //is_converged=estimate_nl_error();//l
            deviation=(temp_locally_relevant_solution_P.l2_norm()
                       - old_temp_locally_relevant_solution_P.l2_norm() )
                       /  temp_locally_relevant_solution_P.l2_norm();
            pcout<<"deviation: "<<deviation<<std::endl;

            if(std::abs(deviation)<tolerance)
              {
                is_converged=true;
              }
            nl_loop_count++;


          }//end nonlinear loop
        if(nl_loop_count>maxiter)
          {
            pcout<<"not converged";

          }
        locally_relevant_solution_P=temp_locally_relevant_solution_P;
        locally_relevant_solution_Sigma=temp_locally_relevant_solution_Sigma;




        // Solve temperature (coefficients depend on porosity, and TODO: should influence viscosity)
        if (compute_temperature)
          {
          pcout <<"TEMPTMEEPRPERPERPEPRPERPE";

            assemble_matrices_T();
            forge_system_T();
           solve_time_step_T();
          }
        setup_system_F();
        assemble_F();
        solve_F();
        //setup_system_F();
//locally_relevant_solution_F = -1*base_sedimentation_rate;
        //    if (get_output && time - (output_number)*output_time > 0)
        //      output_results();
        if (timestep_number % output_interval == 0) {
            display_vectors();

        }

        prepare_next_time_step();
    } // end of time loop

    //output once at the end
   // output_results_pp();
} //end of run function


} // end namespace CPPLS


int main(int argc, char* argv[])
{

    try {
        using namespace dealii;
        using namespace CPPLS;

        auto t0 = std::chrono::high_resolution_clock::now();

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        CPPLS::Parameters parameters;
        parameters.read_parameter_file("parameters.prm");
        CPPLS::MaterialData material_data;
        if (parameters.dimension==2)
        {
            LayerMovementProblem<2> run_layers(parameters, material_data);
            run_layers.run();
        }
        else if (parameters.dimension==3)
          {
            LayerMovementProblem<3> run_layers(parameters, material_data);
            run_layers.run();

          }
        else
          {
             AssertThrow (false, ExcNotImplemented());
          }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
            std::cout << "time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                      << " milliseconds." << std::endl;
        }
    }
    catch (std::exception& exc) {
        std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;

        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}
