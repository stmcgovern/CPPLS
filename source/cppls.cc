

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
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
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

//#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

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

#include "level_set_solver.h"
#include "material_data.h"
#include "my_utility_functions.h"
#include "parameters.h"
#include "physical_functions.h"

namespace CPPLS {
using namespace dealii;

// some free functions

// template <int dim>
// void print_mesh_info(const Triangulation<dim>& triangulation, const std::string& filename)
//{
//  std::cout << "Mesh info:" << std::endl
//            << " dimension: " << dim << std::endl
//            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

//  std::map<unsigned int, unsigned int> boundary_count;
//  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();
//  for (; cell != endc; ++cell) {
//    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
//      if (cell->face(face)->at_boundary())
//        boundary_count[cell->face(face)->boundary_id()]++;
//    }
//  }
//  std::cout << " boundary indicators: ";
//  for (std::map<unsigned int, unsigned int>::iterator it = boundary_count.begin(); it != boundary_count.end(); ++it) {
//    std::cout << it->first << "(" << it->second << " times) ";
//  }
//  std::cout << std::endl;

//  std::ofstream out(filename.c_str());
//  GridOut grid_out;
//  grid_out.write_eps(triangulation, out);
//  std::cout << " written to " << filename << std::endl << std::endl;
//}

// double porosity(const double pressure, const double overburden, const double initial_porosity,
//                const double compaction_coefficient)
//{
//  return (initial_porosity * std::exp(-1 * compaction_coefficient * (overburden - pressure)));
//}
// double permeability(const double porosity, const double initial_permeability, const double initial_porosity)
//{
//  return (initial_permeability * (1 - porosity) / (1 - initial_porosity));
//}

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

  // FE basis spaces

  // pressure
  int degree_P{1};
  DoFHandler<dim> dof_handler_P;
  FE_Q<dim> fe_P;
  IndexSet locally_owned_dofs_P;
  IndexSet locally_relevant_dofs_P;

  // temperature
  int degree_T{1};
  DoFHandler<dim> dof_handler_T;
  FE_Q<dim> fe_T;
  IndexSet locally_owned_dofs_T;
  IndexSet locally_relevant_dofs_T;

  // level set (can multiple level sets use the same of below? probably not IndexSets)std::vector<IndexSets>
  int degree_LS{1};
  DoFHandler<dim> dof_handler_LS;
  FE_Q<dim> fe_LS;
  IndexSet locally_owned_dofs_LS;
  IndexSet locally_relevant_dofs_LS;

  int degree_quad_data{2};
  DoFHandler<dim> dof_handler_Q;
  FE_DGQ<dim> fe_DGQ_Q;
  IndexSet locally_owned_dofs_Q;

  // output stream where only mpi rank 0 output gets to stdout
  ConditionalOStream pcout;

  TimerOutput computing_timer;

  double time_step;
  double current_time;
  double output_number;
  double final_time;
  int timestep_number;

  // set timestepping scheme 1 implicit euler, 1/2 CN, 0 explicit euler
  const double theta;

  ConstraintMatrix constraints_P;
  ConstraintMatrix constraints_T;
  ConstraintMatrix constraints_LS;
  ConstraintMatrix constraints_Q;
  ConstraintMatrix constraints_poisson;

  // FE Field Solution Vectors

  LA::MPI::Vector locally_relevant_solution_LS_0; // ls
  LA::MPI::Vector old_locally_relevant_solution_LS_0;

  LA::MPI::Vector locally_relevant_solution_LS_1;
  LA::MPI::Vector completely_distributed_solution_LS_1;

  LA::MPI::Vector locally_relevant_solution_LS_2;
  LA::MPI::Vector completely_distributed_solution_LS_2;

  LA::MPI::Vector locally_relevant_solution_LS_3;
  LA::MPI::Vector completely_distributed_solution_LS_3;




  LA::MPI::Vector locally_relevant_solution_P;
  LA::MPI::Vector old_locally_relevant_solution_P;
  LA::MPI::Vector locally_relevant_solution_T;
  LA::MPI::Vector old_locally_relevant_solution_T;
  LA::MPI::Vector locally_relevant_solution_Fx;
  LA::MPI::Vector locally_relevant_solution_Fy; // speed function
  LA::MPI::Vector completely_distributed_solution_LS_0;
  LA::MPI::Vector completely_distributed_solution_P;
  LA::MPI::Vector completely_distributed_solution_T;
  LA::MPI::Vector completely_distributed_solution_f;
  LA::MPI::Vector completely_distributed_solution_Fx;
  LA::MPI::Vector completely_distributed_solution_Fy;

  LA::MPI::Vector rhs_P;
  LA::MPI::Vector old_rhs_P;
  LA::MPI::Vector system_rhs_P;

  LA::MPI::Vector rhs_T;
  LA::MPI::Vector old_rhs_T;
  LA::MPI::Vector system_rhs_T;

  // Physical Vectors

  LA::MPI::Vector overburden;
  LA::MPI::Vector old_overburden;

  LA::MPI::Vector bulkdensity;
  LA::MPI::Vector porosity;
  LA::MPI::Vector old_porosity;
  LA::MPI::Vector permeability;

  LA::MPI::Vector bulkheat_capacity;
  LA::MPI::Vector thermal_conductivity;


  // Sparse Matrices
  LA::MPI::SparseMatrix laplace_matrix_P;
  LA::MPI::SparseMatrix mass_matrix_P;
  LA::MPI::SparseMatrix system_matrix_P;

  LA::MPI::SparseMatrix laplace_matrix_T;
  LA::MPI::SparseMatrix mass_matrix_T;
  LA::MPI::SparseMatrix system_matrix_T;


  LA::MPI::Vector poisson_solution;
  LA::MPI::Vector poisson_rhs;
  LA::MPI::SparseMatrix poisson_system_matrix;


  // for boundary conditions
  std::vector<unsigned int> boundary_values_id_LS;

  std::vector<double> boundary_values_LS;

  // Member Functions

  // create mesh
  void setup_geometry();

  // create appropriately sized vectors and matrices
  void setup_system_P();
  void setup_system_T();
  void setup_system_LS();
  void setup_system_Q();
  void initial_conditions();
  void set_boundary_inlet();
  void get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
                              std::vector<double>& boundary_values_LS);

  void setup_material_configuration();
  // initialize vectors


  void assemble_matrices_P();
  void forge_system_P();
  void solve_time_step_P();

  void assemble_matrices_T();
  void forge_system_T();
  void solve_time_step_T();

  void compute_bulkdensity();
  void compute_overburden();
  void compute_porosity_and_permeability();
  void compute_speed_function();

  void setup_speed_function_poisson();
  void solve_poisson();

  void prepare_next_time_step();

  void output_vectors_LS();
  void output_vectors_P();
  void output_vectors_T();
  void output_vectors_Q();
  void display_vectors()
  {
    output_vectors_LS();
    output_vectors_P();
    output_vectors_Q();
    output_vectors_T();
    output_number++;
  }
};

// Constructor

template <int dim>
LayerMovementProblem<dim>::LayerMovementProblem(const CPPLS::Parameters& parameters,
                                                const CPPLS::MaterialData& material_data)
  : parameters(parameters)
  , material_data(material_data)
  , mpi_communicator(MPI_COMM_WORLD)
  , n_mpi_processes{Utilities::MPI::n_mpi_processes(mpi_communicator)}
  , this_mpi_process{Utilities::MPI::this_mpi_process(mpi_communicator)}
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement |
                                                             Triangulation<dim>::smoothing_on_coarsening))
  , fe_P(degree_P)
  , fe_T(degree_T)
  , fe_LS(degree_LS)
  , fe_DGQ_Q(degree_quad_data)
  , dof_handler_P(triangulation)
  , dof_handler_T(triangulation)
  , dof_handler_LS(triangulation)
  , dof_handler_Q(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
  , time_step((parameters.stop_time - parameters.start_time) / parameters.n_time_steps)
  , current_time{parameters.start_time}
  , final_time{parameters.stop_time}
  , output_number{0}
  , theta(parameters.theta)

        {};

// Destructor
template <int dim>
LayerMovementProblem<dim>::~LayerMovementProblem()
{
  dof_handler_P.clear();
  dof_handler_T.clear();
  dof_handler_LS.clear();
  dof_handler_Q.clear();
}

//
template <int dim>
void LayerMovementProblem<dim>::setup_geometry()
{
  TimerOutput::Scope t(computing_timer, "setup_geometry");
  GridGenerator::hyper_cube(triangulation, 0, parameters.box_size, true);
  // GridGenerator::subdivided_hyper_rectangle(triangulation, 0, parameters.box_size);
  triangulation.refine_global(parameters.initial_refinement_level);
  //print_mesh_info(triangulation, "my_grid");
  for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    cell->set_material_id(0);
  }
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_P()
{
  TimerOutput::Scope t(computing_timer, "setup_system_P");

  dof_handler_P.distribute_dofs(fe_P);

  pcout << std::endl
        << "============Pressure===============" << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_P.n_dofs() << std::endl
        << std::endl;

  locally_owned_dofs_P = dof_handler_P.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_P, locally_relevant_dofs_P);

  // vector setup
  locally_relevant_solution_P.reinit(locally_owned_dofs_P, locally_relevant_dofs_P, mpi_communicator);
  locally_relevant_solution_P = 0;

  completely_distributed_solution_P.reinit(locally_owned_dofs_P, mpi_communicator);

  old_locally_relevant_solution_P.reinit(locally_owned_dofs_P, locally_relevant_dofs_P, mpi_communicator);
  old_locally_relevant_solution_P = 0;
  rhs_P.reinit(locally_owned_dofs_P, mpi_communicator);
  rhs_P = 0;
  old_rhs_P.reinit(locally_owned_dofs_P, mpi_communicator);
  old_rhs_P = 0;

  system_rhs_P.reinit(locally_owned_dofs_P, mpi_communicator);

  // constraints

  constraints_P.clear();

  constraints_P.reinit(locally_relevant_dofs_P);

  DoFTools::make_hanging_node_constraints(dof_handler_P, constraints_P);
  // zero dirichlet at top
  VectorTools::interpolate_boundary_values(dof_handler_P, 3, ZeroFunction<dim>(), constraints_P);
  constraints_P.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs_P);

  DoFTools::make_sparsity_pattern(dof_handler_P, dsp, constraints_P, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler_P.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs_P);
  // setup matrices

  system_matrix_P.reinit(locally_owned_dofs_P, locally_owned_dofs_P, dsp, mpi_communicator);
  laplace_matrix_P.reinit(locally_owned_dofs_P, locally_owned_dofs_P, dsp, mpi_communicator);
  mass_matrix_P.reinit(locally_owned_dofs_P, locally_owned_dofs_P, dsp, mpi_communicator);
}


template <int dim>
void LayerMovementProblem<dim>::setup_system_T()
{
  TimerOutput::Scope t(computing_timer, "setup_system_T");

  dof_handler_T.distribute_dofs(fe_T);

  pcout << std::endl
        << "============Temperature===============" << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_T.n_dofs() << std::endl
        << std::endl;

  locally_owned_dofs_T = dof_handler_T.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_T, locally_relevant_dofs_T);

  // vector setup
  locally_relevant_solution_T.reinit(locally_owned_dofs_T, locally_relevant_dofs_T, mpi_communicator);
  locally_relevant_solution_T = 0;

  completely_distributed_solution_T.reinit(locally_owned_dofs_T, mpi_communicator);

  old_locally_relevant_solution_T.reinit(locally_owned_dofs_T, locally_relevant_dofs_T, mpi_communicator);
  old_locally_relevant_solution_T = 0;
  rhs_T.reinit(locally_owned_dofs_T, mpi_communicator);
  rhs_T = 0;
  old_rhs_T.reinit(locally_owned_dofs_T, mpi_communicator);
  old_rhs_T = 0;

  system_rhs_T.reinit(locally_owned_dofs_T, mpi_communicator);

  // constraints

  constraints_T.clear();

  constraints_T.reinit(locally_relevant_dofs_T);

  DoFTools::make_hanging_node_constraints(dof_handler_T, constraints_T);
  // zero dirichlet at top
  VectorTools::interpolate_boundary_values(dof_handler_T, 3, ZeroFunction<dim>(), constraints_T);
  //Keep top at fixed temperature, TODO check compatibility condition
  //VectorTools::interpolate_boundary_values(dof_handler_T, 3, ConstantFunction<dim>(20), constraints_T);
  constraints_T.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs_T);

  DoFTools::make_sparsity_pattern(dof_handler_T, dsp, constraints_T, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler_T.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs_T);
  // setup matrices

  system_matrix_T.reinit(locally_owned_dofs_T, locally_owned_dofs_T, dsp, mpi_communicator);
  laplace_matrix_T.reinit(locally_owned_dofs_T, locally_owned_dofs_T, dsp, mpi_communicator);
  mass_matrix_T.reinit(locally_owned_dofs_T, locally_owned_dofs_T, dsp, mpi_communicator);
}



template <int dim>
void LayerMovementProblem<dim>::setup_system_Q()
{
  TimerOutput::Scope t(computing_timer, "setup_system_Q");

  dof_handler_Q.distribute_dofs(fe_DGQ_Q);

  pcout << std::endl
        << "============Quad point data===============" << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_Q.n_dofs() << std::endl
        << std::endl;

  locally_owned_dofs_Q = dof_handler_Q.locally_owned_dofs();
  // DoFTools::extract_locally_relevant_dofs(dof_handler_P, locally_relevant_dofs_P);

  // initialize
  porosity.reinit(locally_owned_dofs_Q, mpi_communicator);
  old_porosity.reinit(locally_owned_dofs_Q, mpi_communicator);
  old_porosity = 0.61; // initial basement value TODO put in parameter file
  overburden.reinit(locally_owned_dofs_Q, mpi_communicator);
  old_overburden.reinit(locally_owned_dofs_Q, mpi_communicator);
  bulkdensity.reinit(locally_owned_dofs_Q, mpi_communicator);

  bulkheat_capacity.reinit(locally_owned_dofs_Q, mpi_communicator);
  thermal_conductivity.reinit(locally_owned_dofs_Q, mpi_communicator);

  permeability.reinit(locally_owned_dofs_Q, mpi_communicator);
  permeability = 0;
  porosity = 0.61;

  DoFTools::make_hanging_node_constraints(dof_handler_Q, constraints_Q);
  constraints_Q.close();
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_LS()
{
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
  locally_relevant_solution_LS_0 = 0;

  locally_relevant_solution_LS_1.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  locally_relevant_solution_LS_1 = 0;

  locally_relevant_solution_LS_2.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  locally_relevant_solution_LS_2 = 0;


  locally_relevant_solution_LS_3.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  locally_relevant_solution_LS_3 = 0;


  completely_distributed_solution_LS_0.reinit(locally_owned_dofs_LS, mpi_communicator);
 completely_distributed_solution_LS_1.reinit(locally_owned_dofs_LS, mpi_communicator);
  completely_distributed_solution_LS_2.reinit(locally_owned_dofs_LS, mpi_communicator);
   completely_distributed_solution_LS_3.reinit(locally_owned_dofs_LS, mpi_communicator);

  // ghosted
  locally_relevant_solution_Fx.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  locally_relevant_solution_Fx = 0;

  completely_distributed_solution_Fx.reinit(locally_owned_dofs_LS, mpi_communicator);

  locally_relevant_solution_Fy.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  locally_relevant_solution_Fy = 0;
  completely_distributed_solution_Fy.reinit(locally_owned_dofs_LS, mpi_communicator);

  // constraints

  constraints_LS.clear();

  constraints_LS.reinit(locally_relevant_dofs_LS);

  DoFTools::make_hanging_node_constraints(dof_handler_LS, constraints_LS);

  constraints_LS.close();

  //  // create sparsity pattern

  //  DynamicSparsityPattern dsp(locally_relevant_dofs_LS);

  //  DoFTools::make_sparsity_pattern(dof_handler_LS, dsp, constraints_LS, false);
  //  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler_LS.n_locally_owned_dofs_per_processor(),
  //  mpi_communicator,
  //                                             locally_relevant_dofs_LS);
}

template <int dim>
void LayerMovementProblem<dim>::initial_conditions()
{
  // Precondition: the non/ghosted vectors have been initialized, and constraints closed

  // init condition for P
  completely_distributed_solution_P = 0;
  VectorTools::interpolate_boundary_values(dof_handler_P, /*top boundary*/ 3, ZeroFunction<dim>(), constraints_P);
  constraints_P.distribute(completely_distributed_solution_P);
  locally_relevant_solution_P = completely_distributed_solution_P;


  // init condition for T   //TODO
  completely_distributed_solution_T = 0;
  VectorTools::interpolate_boundary_values(dof_handler_T, /*top boundary*/ 3, ZeroFunction<dim>(), constraints_T);
  constraints_T.distribute(completely_distributed_solution_T);
  locally_relevant_solution_T = completely_distributed_solution_T;

  // init condition for LS
  //all the others will share this
  completely_distributed_solution_LS_0 = 0;
  VectorTools::interpolate(dof_handler_LS, Initial_LS<dim>(), completely_distributed_solution_LS_0);
  constraints_LS.distribute(completely_distributed_solution_LS_0);
  locally_relevant_solution_LS_0 = completely_distributed_solution_LS_0;
}

// template <int dim>
// void LayerMovementProblem<dim>::set_boundary_inlet()
//{
//  const QGauss<dim - 1> face_quadrature_formula(1); // center of the face
//  FEFaceValues<dim> fe_face_values(fe_LS, face_quadrature_formula,
//                                   update_values | update_quadrature_points | update_normal_vectors);
//  const unsigned int n_face_q_points = face_quadrature_formula.size();
//  std::vector<double> u_value(n_face_q_points);
//  std::vector<double> v_value(n_face_q_points);

//  typename DoFHandler<dim>::active_cell_iterator cell_U = dof_handler_LS.begin_active(), endc_U =
//  dof_handler_LS.end(); Tensor<1, dim> u;

//  for (; cell_U != endc_U; ++cell_U)
//    if (cell_U->is_locally_owned())
//      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//        if (cell_U->face(face)->at_boundary()) {
//          fe_face_values.reinit(cell_U, face);
//          // fe_face_values.get_function_values(locally_relevant_solution_u,u_value);
//          // fe_face_values.get_function_values(locally_relevant_solution_v,v_value);
//          u[0] = 0; // u_value[0];
//          u[1] = 0; // v_value[0];
//          u[2] = 0.1;
//          if (fe_face_values.normal_vector(0) * u < -1e-14)
//            cell_U->face(face)->set_boundary_id(10); // SET ID 10 to inlet BOUNDARY (10 is an arbitrary number)
//        }
//}

template <int dim>
void LayerMovementProblem<dim>::get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
                                                       std::vector<double>& boundary_values_LS)
{
  std::map<unsigned int, double> map_boundary_values_LS;
  unsigned int boundary_id = 0;

  // set_boundary_inlet();
  boundary_id = 10; // inlet
  // we define the inlet to be at the top, i.e. boundary_id=3
  VectorTools::interpolate_boundary_values(dof_handler_LS, 3, BoundaryPhi<dim>(1.0), map_boundary_values_LS);
  boundary_values_id_LS.resize(map_boundary_values_LS.size());
  boundary_values_LS.resize(map_boundary_values_LS.size());
  std::map<unsigned int, double>::const_iterator boundary_value_LS = map_boundary_values_LS.begin();
  for (int i = 0; boundary_value_LS != map_boundary_values_LS.end(); ++boundary_value_LS, ++i) {
    boundary_values_id_LS[i] = boundary_value_LS->first;
    boundary_values_LS[i] = boundary_value_LS->second;
  }
}

template <int dim>
void LayerMovementProblem<dim>::compute_bulkdensity()
{
  TimerOutput::Scope t(computing_timer, "compute_bulkdensity");

  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe_DGQ_Q, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe_DGQ_Q.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  // FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  // Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<double> bulkdensity_at_quad(n_q_points);
  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> bulkheat_capacity_at_quad(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell : filter_iterators(dof_handler_Q.active_cell_iterators(),
                                    IteratorFilters::LocallyOwnedCell()))
    {
    fe_values.reinit(cell);

    const double rock_density = material_data.get_solid_density(cell->material_id());
    const double rock_heat_capacity = material_data.get_heat_capacity(cell->material_id());

    fe_values.get_function_values(porosity, porosity_at_quad);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      //        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      //          for (unsigned int j = 0; j < dofs_per_cell; ++j) {

      bulkdensity_at_quad[q_point] =
          porosity_at_quad[q_point] * material_data.fluid_density + (1 - porosity_at_quad[q_point]) * rock_density;

      bulkheat_capacity_at_quad[q_point] =
          porosity_at_quad[q_point] * material_data.fluid_heat_capacity + (1 - porosity_at_quad[q_point]) * rock_heat_capacity;

    }

    cell->get_dof_indices(local_dof_indices);
    constraints_Q.distribute_local_to_global(bulkdensity_at_quad, local_dof_indices, bulkdensity);
    constraints_Q.distribute_local_to_global(bulkheat_capacity_at_quad, local_dof_indices, bulkheat_capacity);

  }
  bulkdensity.compress(VectorOperation::add);
  bulkheat_capacity.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::compute_overburden()
{
  TimerOutput::Scope t(computing_timer, "compute_overburden");

  old_overburden = overburden;

  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe_DGQ_Q, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe_DGQ_Q.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> bulkdensity_at_quad(n_q_points);

  Point<dim> point_for_depth;
  //TODO: this does not take into account variable bulkdensity in a column
  // this very simply takes the bulkdensity at a point and multiplies by depth
  for (auto cell : filter_iterators(dof_handler_Q.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

    fe_values.reinit(cell);

    // fe_values.get_function_values(interface_LS, phi_at_quad);
    fe_values.get_function_values(bulkdensity, bulkdensity_at_quad);
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      point_for_depth = fe_values.quadrature_point(q_point);
      // TODO: make dim independent p[1] p[2]
      //TODO: make spatial scale consistent , ie. 1 -> box.size
      overburden_at_quad[q_point] = /* material_data.g */ 9.81 * (1 - point_for_depth[1]) * bulkdensity_at_quad[q_point];
    }
    cell->get_dof_indices(local_dof_indices);
    constraints_Q.distribute_local_to_global(overburden_at_quad, local_dof_indices, overburden);
  }
  overburden.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::compute_porosity_and_permeability()
{
  TimerOutput::Scope t(computing_timer, "compute_porosity and permeability");

  old_porosity = porosity;
  porosity=0;
  permeability=0;

  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values_Q(fe_DGQ_Q, quadrature_formula, update_values | update_quadrature_points);
  FEValues<dim> fe_values_P(fe_P, quadrature_formula, update_values | update_quadrature_points);
  const unsigned int dofs_per_cell = fe_DGQ_Q.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> pressure_at_quad(n_q_points);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> permeability_at_quad(n_q_points);
  std::vector<double> thermal_conductivity_at_quad(n_q_points);

  // use the pre c++11 notation, since we are iterating using two dof_handlers

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_P.begin_active(), endc = dof_handler_P.end();
  typename DoFHandler<dim>::active_cell_iterator cell_Q = dof_handler_Q.begin_active();

  // for (auto cell : filter_iterators(dof_handler_Q.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
  for (; cell != endc; ++cell, ++cell_Q) {
    // assert that cell=cell_Q
    // Assert (cell->id() == cell_Q->id());
    if (cell->is_locally_owned()) {
      fe_values_Q.reinit(cell_Q);
      fe_values_P.reinit(cell);
      const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
      const double initial_permeability = material_data.get_surface_permeability(cell->material_id());

      const double compaction_coefficient = material_data.get_compressibility_coefficient(cell->material_id());

      fe_values_P.get_function_values(locally_relevant_solution_P, pressure_at_quad);
      fe_values_Q.get_function_values(overburden, overburden_at_quad);
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

        porosity_at_quad[q_point] = CPPLS::porosity(pressure_at_quad[q_point], overburden_at_quad[q_point],
                                                    initial_porosity, compaction_coefficient);
        permeability_at_quad[q_point] =
            CPPLS::permeability(porosity_at_quad[q_point], initial_permeability, initial_porosity);
        //TODO put a real function in here
        thermal_conductivity_at_quad[q_point]= 1;
           // CPPLS::thermal_conductivity(porosity_at_quad[q_point],);


        Assert (porosity_at_quad[q_point] <1, ExcInternalError()) ;
        Assert (permeability_at_quad[q_point] >0, ExcInternalError()) ;
      }
      cell_Q->get_dof_indices(local_dof_indices);
      constraints_Q.distribute_local_to_global(porosity_at_quad, local_dof_indices, porosity);
      constraints_Q.distribute_local_to_global(permeability_at_quad, local_dof_indices, permeability);
      constraints_Q.distribute_local_to_global(thermal_conductivity_at_quad, local_dof_indices, thermal_conductivity);


    } // local
  }   // cell loop
  porosity.compress(VectorOperation::add);
  permeability.compress(VectorOperation::add);
  thermal_conductivity.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::setup_material_configuration()
{
  TimerOutput::Scope t(computing_timer, "set_material_configuration");
  // This function sets material ids of cells based on the location of the interface, i.e. loc_rel_solution_LS

  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe_LS, quadrature_formula, update_values | update_quadrature_points);

  //  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  //  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> LS0_at_quad(n_q_points);
  std::vector<double> LS1_at_quad(n_q_points);
   std::vector<double> LS2_at_quad(n_q_points);
    std::vector<double> LS3_at_quad(n_q_points);

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

  //std::vector<double> id_sum(5);
  double id_sum0{0};
  double id_sum1{0};
  double id_sum2{0};
  double id_sum3{0};
  //double id_sum4{0};

  for (auto cell : filter_iterators(dof_handler_LS.active_cell_iterators(),
                                    IteratorFilters::LocallyOwnedCell())) {
    id_sum0=0;
    id_sum1=0;
    id_sum2=0;
    id_sum3=0;
    //id_sum4=0;
    fe_values.reinit(cell);
    fe_values.get_function_values(locally_relevant_solution_LS_0, LS0_at_quad);
    fe_values.get_function_values(locally_relevant_solution_LS_1, LS1_at_quad);
    fe_values.get_function_values(locally_relevant_solution_LS_2, LS2_at_quad);
    fe_values.get_function_values(locally_relevant_solution_LS_3, LS3_at_quad);
    //fe_values.get_function_values(locally_relevant_solution_LS_4, LS4_at_quad);


    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      id_sum0 += LS0_at_quad[q_point];
      id_sum1 += LS1_at_quad[q_point];
      id_sum2 += LS2_at_quad[q_point];
      id_sum3 += LS3_at_quad[q_point];
//      id_sum4 += LS4_at_quad[q_point];

    }
    if (id_sum0 <= 0){cell->set_material_id(0);}
    else if (id_sum1 <= 0){cell->set_material_id(1);}
    else if (id_sum2 <= 0){cell->set_material_id(2);}
    else if (id_sum3 <= 0){cell->set_material_id(3);}
    else {
        pcout<<"bad level set counting";
      Assert(false, ExcNotImplemented());
    }
  }// end cell loop
}

template <int dim>
void LayerMovementProblem<dim>::assemble_matrices_P()
{
  TimerOutput::Scope t(computing_timer, "assembly_P");
  const QGauss<dim> quadrature_formula(3);

  // RightHandSide<dim> right_hand_side;
  // //set time too
  // right_hand_side.set_time(time);
  // DiffusionCoefficient<dim> diffusion_coeff;

  FEValues<dim> fe_values(fe_P, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEValues<dim> fe_values_Q(fe_DGQ_Q, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe_P.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // std::vector<double> u_at_quad(n_q_points);
  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> old_overburden_at_quad(n_q_points);
  std::vector<double> permeability_at_quad(n_q_points);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_P.begin_active(), endc = dof_handler_P.end();
  typename DoFHandler<dim>::active_cell_iterator cell_Q = dof_handler_Q.begin_active();

  // for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
  for (; cell != endc; ++cell, ++cell_Q) {
    // assert that cell=cell_Q
    // Assert
    if (cell->is_locally_owned()) {
      cell_laplace_matrix = 0;
      cell_mass_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);
      fe_values_Q.reinit(cell_Q);

      fe_values_Q.get_function_values(porosity, porosity_at_quad);
      fe_values_Q.get_function_values(permeability, permeability_at_quad);
      fe_values_Q.get_function_values(overburden, overburden_at_quad);
      fe_values_Q.get_function_values(old_overburden, old_overburden_at_quad);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const double diff_coeff_at_quad =
            (permeability_at_quad[q_point] / (porosity_at_quad[q_point] * material_data.fluid_viscosity));
        const double rhs_at_quad = (overburden_at_quad[q_point] - old_overburden_at_quad[q_point] ) / time_step /*- 9.8*material_data.fluid_density*0.08*/;

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {

            cell_laplace_matrix(i, j) +=
                diff_coeff_at_quad *
                (fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

            //                    cell_laplace_matrix(i,j) += diffusion_coeff.value(fe_values.quadrature_point
            //                    (q_point)) *
            //                                        (fe_values.shape_grad(i,q_point) *
            //                                         fe_values.shape_grad(j,q_point) *
            //                                         fe_values.JxW(q_point));

            cell_mass_matrix(i, j) +=
                (fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));

            //          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_point)) *
            //                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
             cell_rhs(i) += (rhs_at_quad * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints_P.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_P,
                                               rhs_P);
      constraints_P.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_P);
    }
  }

  // Notice that the assembling above is just a local operation. So, to
  // form the "global" linear system, a synchronization between all
  // processors is needed. This could be done by invoking the function
  // compress(). See @ref GlossCompress  "Compressing distributed objects"
  // for more information on what is compress() designed to do.
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

  tmp.reinit(locally_owned_dofs_P, locally_relevant_dofs_P, mpi_communicator);
  //    forcing_terms.reinit (locally_owned_dofs,
  //                          locally_relevant_dofs, mpi_communicator);
  forcing_terms.reinit(locally_owned_dofs_P, mpi_communicator);

  old_locally_relevant_solution_P = locally_relevant_solution_P;
  mass_matrix_P.vmult(system_rhs_P, old_locally_relevant_solution_P);

  laplace_matrix_P.vmult(tmp, old_locally_relevant_solution_P);
  //  pcout << "laplace symmetric: " << laplace_matrix.is_symmetric()<<std::endl;
  system_rhs_P.add(-(1 - theta) * time_step, tmp);

  forcing_terms.add(time_step * theta, rhs_P);

  forcing_terms.add(time_step * (1 - theta), old_rhs_P);

  system_rhs_P += forcing_terms;
  // system_matrix.compress (VectorOperation::add);

  system_matrix_P.copy_from(mass_matrix_P);
  // system_matrix.compress (VectorOperation::add);

  system_matrix_P.add(laplace_matrix_P, time_step * (1 - theta));

  // constraints.condense (system_matrix, system_rhs);

  // There is one more operation we need to do before we
  // can solve it: boundary values. To this end, we create
  // a boundary value object, set the proper time to the one
  // of the current time step, and evaluate it as we have
  // done many times before. The result is used to also
  // set the correct boundary values in the linear system:
  //            {
  //              BoundaryValues<dim> boundary_values_function;
  //              boundary_values_function.set_time(time);

  //              std::map<types::global_dof_index, double> boundary_values;
  //              VectorTools::interpolate_boundary_values(dof_handler,
  //                                                       0,
  //                                                       boundary_values_function,
  //                                                       boundary_values);

  //    //          MatrixTools::apply_boundary_values(boundary_values,
  //    //                                             system_matrix,
  //    //                                             locally_relevant_solution,
  //    //                                             system_rhs);
  //            }
}

template <int dim>
void LayerMovementProblem<dim>::solve_time_step_P()
{
  TimerOutput::Scope t(computing_timer, "solve_time_step_P");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs_P, mpi_communicator);

  SolverControl solver_control(dof_handler_P.n_dofs(), 1e-12);
  LA::SolverCG solver(solver_control, mpi_communicator);

  LA::MPI::PreconditionAMG preconditioner;

  LA::MPI::PreconditionAMG::AdditionalData data;

  data.symmetric_operator = true;
  preconditioner.initialize(system_matrix_P, data);

  solver.solve(system_matrix_P, completely_distributed_solution, system_rhs_P, preconditioner);

  pcout << " Pressure system solved in " << solver_control.last_step() << " iterations." << std::endl;

  constraints_P.distribute(completely_distributed_solution);

  locally_relevant_solution_P = completely_distributed_solution;
}

//TODO test the temperature part

template <int dim>
void LayerMovementProblem<dim>::assemble_matrices_T()
{
  TimerOutput::Scope t(computing_timer, "assembly_T");
  const QGauss<dim> quadrature_formula(3);
  const QGauss<dim-1> face_quadrature_formula(3);


  FEValues<dim> fe_values(fe_T, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe_T, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);


  FEValues<dim> fe_values_Q(fe_DGQ_Q, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe_T.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> bulkheat_capacity_at_quad(n_q_points);
  std::vector<double> bulkdensity_at_quad(n_q_points);
  std::vector<double> thermal_conductivity_at_quad(n_q_points);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_T.begin_active(), endc = dof_handler_T.end();
  typename DoFHandler<dim>::active_cell_iterator cell_Q = dof_handler_Q.begin_active();

  // for (auto cell : filter_iterators(dof_handler_T.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
  for (; cell != endc; ++cell, ++cell_Q) {
    // assert that cell=cell_Q
    // Assert
    if (cell->is_locally_owned()) {
      cell_laplace_matrix = 0;
      cell_mass_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);
      fe_values_Q.reinit(cell_Q);

      fe_values_Q.get_function_values(porosity, porosity_at_quad);
      fe_values_Q.get_function_values(thermal_conductivity, thermal_conductivity_at_quad);
      fe_values_Q.get_function_values(bulkdensity, bulkdensity_at_quad);
      fe_values_Q.get_function_values(bulkheat_capacity, bulkheat_capacity_at_quad);
  //TODO
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const double diff_coeff_at_quad =10;
          //  thermal_conductivity_at_quad[q_point] /(bulkheat_capacity_at_quad[q_point]*bulkdensity_at_quad[q_point] );
        const double rhs_at_quad =0; //TODO bottom boundary flux from parameter file

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {

            cell_laplace_matrix(i, j) +=
                diff_coeff_at_quad *
                (fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

            //                    cell_laplace_matrix(i,j) += diffusion_coeff.value(fe_values.quadrature_point
            //                    (q_point)) *
            //                                        (fe_values.shape_grad(i,q_point) *
            //                                         fe_values.shape_grad(j,q_point) *
            //                                         fe_values.JxW(q_point));

            cell_mass_matrix(i, j) +=
                (fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));

            //          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_point)) *
            //                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
             cell_rhs(i) += (rhs_at_quad * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
          } //end j
        } //end i
      }//end q

      for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
        if (cell->face(face_number)->at_boundary()
            &&
            (cell->face(face_number)->boundary_id() == 2)) //bottom of domain TODO
          {
            fe_face_values.reinit (cell, face_number);
            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
              {
//                const double neumann_value
//                  = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
//                     fe_face_values.normal_vector(q_point));
                //TODO pick the right flux value
                const double neumann_value=100; //represents the bottom flux boundary condition
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                  cell_rhs(i) += (neumann_value *
                                  fe_face_values.shape_value(i,q_point) *
                                  fe_face_values.JxW(q_point));
                  }
              }
          }
        }// end face loop


      cell->get_dof_indices(local_dof_indices);
      constraints_T.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_T,
                                               rhs_T);
      constraints_T.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_T);
    }
  }

  // Notice that the assembling above is just a local operation. So, to
  // form the "global" linear system, a synchronization between all
  // processors is needed. This could be done by invoking the function
  // compress(). See @ref GlossCompress  "Compressing distributed objects"
  // for more information on what is compress() designed to do.
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

  tmp.reinit(locally_owned_dofs_T, locally_relevant_dofs_T, mpi_communicator);
  //    forcing_terms.reinit (locally_owned_dofs,
  //                          locally_relevant_dofs, mpi_communicator);
  forcing_terms.reinit(locally_owned_dofs_T, mpi_communicator);

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

  // constraints.condense (system_matrix, system_rhs);

  // There is one more operation we need to do before we
  // can solve it: boundary values. To this end, we create
  // a boundary value object, set the proper time to the one
  // of the current time step, and evaluate it as we have
  // done many times before. The result is used to also
  // set the correct boundary values in the linear system:
  //            {
  //              BoundaryValues<dim> boundary_values_function;
  //              boundary_values_function.set_time(time);

  //              std::map<types::global_dof_index, double> boundary_values;
  //              VectorTools::interpolate_boundary_values(dof_handler,
  //                                                       0,
  //                                                       boundary_values_function,
  //                                                       boundary_values);

  //    //          MatrixTools::apply_boundary_values(boundary_values,
  //    //                                             system_matrix,
  //    //                                             locally_relevant_solution,
  //    //                                             system_rhs);
  //            }
}

template <int dim>
void LayerMovementProblem<dim>::solve_time_step_T()
{
  TimerOutput::Scope t(computing_timer, "solve_time_step_T");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs_T, mpi_communicator);

  SolverControl solver_control(dof_handler_T.n_dofs(), 1e-12);
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
void LayerMovementProblem<dim>::setup_speed_function_poisson()
{
  //we use the fe space of P for now

  poisson_solution.reinit(locally_owned_dofs_P, locally_relevant_dofs_P, mpi_communicator);

  poisson_rhs.reinit(locally_owned_dofs_P, mpi_communicator);


  // constraints

  constraints_poisson.clear();

  constraints_poisson.reinit(locally_relevant_dofs_P);

  DoFTools::make_hanging_node_constraints(dof_handler_P, constraints_poisson);
  // zero dirichlet at top
  VectorTools::interpolate_boundary_values(dof_handler_P, 3, ConstantFunction<dim>(-0.2), constraints_poisson);
  constraints_poisson.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs_P);

  DoFTools::make_sparsity_pattern(dof_handler_P, dsp, constraints_poisson, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler_P.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs_P);
  // setup matrices

  poisson_system_matrix.reinit(locally_owned_dofs_P, locally_owned_dofs_P, dsp, mpi_communicator);

}


template <int dim>
void LayerMovementProblem<dim>::solve_poisson()
{
  TimerOutput::Scope t(computing_timer, "solve_poisson");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs_P, mpi_communicator);

  SolverControl solver_control(dof_handler_P.n_dofs(), 1e-12);
  LA::SolverCG solver(solver_control, mpi_communicator);

  LA::MPI::PreconditionAMG preconditioner;

  LA::MPI::PreconditionAMG::AdditionalData data;

  data.symmetric_operator = true;
  preconditioner.initialize(system_matrix_P, data);

  solver.solve(poisson_system_matrix, completely_distributed_solution, poisson_rhs, preconditioner);

  pcout << " speed function poisson system solved in " << solver_control.last_step() << " iterations." << std::endl;

  constraints_P.distribute(completely_distributed_solution);

  poisson_solution = completely_distributed_solution;
}


template <int dim>
void LayerMovementProblem<dim>::compute_speed_function()
{
  TimerOutput::Scope t(computing_timer, "compute_speed_function");

  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values_P(fe_P, quadrature_formula, update_values | update_quadrature_points | update_JxW_values | update_gradients);
  FEValues<dim> fe_values_Q(fe_DGQ_Q, quadrature_formula, update_values | update_quadrature_points | update_gradients);

  const unsigned int dofs_per_cell = fe_P.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> old_porosity_at_quad(n_q_points);
  std::vector<Tensor<1, dim> > grad_porosity_at_quad(n_q_points);
  std::vector<Tensor<1, dim> > grad_old_porosity_at_quad(n_q_points);
  std::vector<double> rhs_at_quad(n_q_points);



  //  std::vector<double> Fx_at_quad(n_q_points);
  //  std::vector<double> Fy_at_quad(n_q_points);

  Vector<double> cell_rhs(dofs_per_cell);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  // std::vector<double> cell_vector_x(dofs_per_cell); TODO dim=3

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_P.begin_active(), endc = dof_handler_P.end();
  typename DoFHandler<dim>::active_cell_iterator cell_Q = dof_handler_Q.begin_active();

  completely_distributed_solution_Fy = 0;
  poisson_rhs=0;
  poisson_system_matrix=0;
  // for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
  for (; cell != endc; ++cell, ++cell_Q) {
    // assert that cell=cell_Q
    // Assert
    if (cell->is_locally_owned()) {

      //  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell()))
      //  {

      fe_values_P.reinit(cell);
      fe_values_Q.reinit(cell_Q);

      // fe_values.get_function_values(interface_LS, phi_at_quad);
      fe_values_Q.get_function_values(porosity, porosity_at_quad);
      fe_values_Q.get_function_values(old_porosity, old_porosity_at_quad);
      fe_values_Q.get_function_gradients(porosity, grad_porosity_at_quad);
      fe_values_Q.get_function_gradients(old_porosity, grad_old_porosity_at_quad);

      cell_rhs=0;
      cell_matrix=0;


      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const double phi=porosity_at_quad[q_point];
          const double dphidz=-1*std::abs(grad_porosity_at_quad[q_point][1]);
          const double old_dphidz=-1*std::abs(grad_old_porosity_at_quad[q_point][1]);
          const double dphidt=(porosity_at_quad[q_point]-old_porosity_at_quad[q_point])/time_step;
          rhs_at_quad[q_point]=//0.2;//grad_porosity_at_quad[q_point][0];
                               -1*( std::pow((1-phi), -2)*(-1*dphidz)*dphidt +
                                std::pow((1-phi), -1)*(dphidz-old_dphidz)/time_step);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {

              cell_matrix(i, j) +=
                  (fe_values_P.shape_grad(i, q_point) * fe_values_P.shape_grad(j, q_point) * fe_values_P.JxW(q_point));

              //          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_point)) *
              //                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
               cell_rhs(i) += (rhs_at_quad[q_point] * fe_values_P.shape_value(i, q_point) * fe_values_P.JxW(q_point));
            } //end j
          } //end i
        }//end q


      cell->get_dof_indices(local_dof_indices);//distribute to correct globally numbered vector

      constraints_poisson.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, poisson_system_matrix, poisson_rhs);
      //constraints_P.distribute_local_to_global(cell_matrix, local_dof_indices, poisson_system_matrix);
    }
    }//end cell loop

  poisson_rhs.compress(VectorOperation::add);
  poisson_system_matrix.compress(VectorOperation::add);

  solve_poisson();
  locally_relevant_solution_Fy=poisson_solution;
}

template <int dim>
void LayerMovementProblem<dim>::prepare_next_time_step()
{
  //  old_porosity = porosity;
  //  old_overburden = overburden;
}

template <int dim>
void LayerMovementProblem<dim>::output_vectors_LS()
{
  TimerOutput::Scope t(computing_timer, "output_LS");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_LS);
  data_out.add_data_vector(locally_relevant_solution_LS_0, "LS");
   data_out.add_data_vector(locally_relevant_solution_LS_1, "LS_1");
  data_out.add_data_vector(locally_relevant_solution_LS_2, "LS_2");
  data_out.add_data_vector(locally_relevant_solution_LS_3, "LS_3");
  data_out.add_data_vector(locally_relevant_solution_Fx, "Fx");
  data_out.add_data_vector(locally_relevant_solution_Fy, "Fy");
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

// TODO output material id

template <int dim>
void LayerMovementProblem<dim>::output_vectors_P()
{
  TimerOutput::Scope t(computing_timer, "output_P");
  // output_number++;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_P);
  data_out.add_data_vector(locally_relevant_solution_P, "P");
  data_out.add_data_vector(old_locally_relevant_solution_P, "old_P");
  //data_out.add_data_vector(poisson_rhs, "poisson_rhs");
  // data_out.add_data_vector(locally_relevant_solution_Fy, "Fy");
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename = ("sol_P_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
      filenames.push_back("sol_P_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::ofstream master_output(("sol_P_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}


template <int dim>
void LayerMovementProblem<dim>::output_vectors_T()
{
  TimerOutput::Scope t(computing_timer, "output_T");
  // output_number++;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_T);
  data_out.add_data_vector(locally_relevant_solution_T, "T");
  data_out.add_data_vector(old_locally_relevant_solution_T, "old_T");
  // data_out.add_data_vector(locally_relevant_solution_Fy, "Fy");
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename = ("sol_T_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
      filenames.push_back("sol_T_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::ofstream master_output(("sol_T_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim>
void LayerMovementProblem<dim>::output_vectors_Q()
{
  TimerOutput::Scope t(computing_timer, "output_Q");
  // output_number++;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_Q);
  data_out.add_data_vector(permeability, "k");
  data_out.add_data_vector(old_porosity, "phi");
  data_out.add_data_vector(overburden, "sigma");

  // data_out.add_data_vector(locally_relevant_solution_Fy, "Fy");
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i) {
    subdomain(i) = triangulation.locally_owned_subdomain();
  }

  data_out.add_data_vector(subdomain, "subdomain");
  LA::MPI::Vector material_kind;
  material_kind.reinit(locally_owned_dofs_Q,  mpi_communicator );
  material_kind=0;

  int i = 0;
  for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    material_kind(i) = static_cast<float>(cell->material_id());
    ++i;
  }
  data_out.add_data_vector(material_kind, "material_kind");

  data_out.build_patches();

  const std::string filename = ("sol_Q_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
      filenames.push_back("sol_Q_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::ofstream master_output(("sol_Q_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim>
void LayerMovementProblem<dim>::run()
{
  // common mesh
  setup_geometry();
  setup_system_P();
  setup_system_T();
  setup_system_Q();
  setup_system_LS();
  setup_speed_function_poisson();

  initial_conditions();
  // display_vectors();

  // at this point the p system and ls system should be ready for the time loop

  // initialize level set solver
  // we use some hardcode defaults for now


  const double min_h = GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2);
  const double cfl = 1;
  const double umax =1;
  time_step=cfl*min_h/umax;


  const double cK = 1.0;
  const double cE = 1.0;
  const bool verbose = true;
  std::string ALGORITHM = "MPP_uH";
  const unsigned int TIME_INTEGRATION = 1; // corresponds to SSP33
  LevelSetSolver<dim> level_set_solver0(degree_LS, degree_P, time_step, cK, cE, verbose, ALGORITHM, TIME_INTEGRATION,
                                       triangulation, mpi_communicator);

  LevelSetSolver<dim> level_set_solver1(degree_LS, degree_P, time_step, cK, cE, verbose, ALGORITHM, TIME_INTEGRATION,
                                       triangulation, mpi_communicator);
  LevelSetSolver<dim> level_set_solver2(degree_LS, degree_P, time_step, cK, cE, verbose, ALGORITHM, TIME_INTEGRATION,
                                       triangulation, mpi_communicator);
  LevelSetSolver<dim> level_set_solver3(degree_LS, degree_P, time_step, cK, cE, verbose, ALGORITHM, TIME_INTEGRATION,
                                       triangulation, mpi_communicator);

  // initialize pressure solver
  // PressureEquation<dim> pressure_solver;
  // initialize temperature solver
  // TemperatureEquation<dim> temperature_solver;

  // BOUNDARY CONDITIONS FOR PHI
  get_boundary_values_LS(boundary_values_id_LS, boundary_values_LS);
  level_set_solver0.set_boundary_conditions(boundary_values_id_LS, boundary_values_LS);
  level_set_solver1.set_boundary_conditions(boundary_values_id_LS, boundary_values_LS);
  level_set_solver2.set_boundary_conditions(boundary_values_id_LS, boundary_values_LS);
  level_set_solver3.set_boundary_conditions(boundary_values_id_LS, boundary_values_LS);

  // set INITIAL CONDITION within TRANSPORT PROBLEM
  level_set_solver0.initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Fx,
                                     locally_relevant_solution_Fy);
  level_set_solver1.initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Fx,
                                     locally_relevant_solution_Fy);
  level_set_solver2.initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Fx,
                                     locally_relevant_solution_Fy);
  level_set_solver3.initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Fx,
                                     locally_relevant_solution_Fy);


  //  // TIME STEPPING
  timestep_number = 1;
  for ( double time = time_step; time <= final_time; time += time_step, ++timestep_number) {
    pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

    // TODO: put the level set solve first in the time loop like Octave

    // set material ids based on locally_relevant_solution_LS
    setup_material_configuration();
    // set physical vectors
    compute_bulkdensity();
    compute_overburden();

    compute_porosity_and_permeability();

    // assemble and solve for pressure
    assemble_matrices_P();

    forge_system_P();
    solve_time_step_P();

    assemble_matrices_T();
    forge_system_T();
    solve_time_step_T();


    compute_speed_function(); // set locally_relevant_solution_u/Fx, locally_relevant_solution_v/Fy

    // Level set computation
    // original level_set_solver.set_velocity(locally_relevant_solution_u, locally_relevant_solution_v);

    {
      TimerOutput::Scope t(computing_timer, "LS");

      level_set_solver0.set_velocity(locally_relevant_solution_Fx, locally_relevant_solution_Fy);
      level_set_solver0.nth_time_step();
      level_set_solver0.get_unp1(locally_relevant_solution_LS_0); // exposes interface vector

    if(timestep_number>50)
      {
        level_set_solver1.set_velocity(locally_relevant_solution_Fx, locally_relevant_solution_Fy);
        level_set_solver1.nth_time_step();
        level_set_solver1.get_unp1(locally_relevant_solution_LS_1); // exposes interface vector

      }
    if(timestep_number>125)
      {
        level_set_solver2.set_velocity(locally_relevant_solution_Fx, locally_relevant_solution_Fy);
        level_set_solver2.nth_time_step();
        level_set_solver2.get_unp1(locally_relevant_solution_LS_2); // exposes interface vector

      }
    if(timestep_number>170)
          {
            level_set_solver3.set_velocity(locally_relevant_solution_Fx, locally_relevant_solution_Fy);
            level_set_solver3.nth_time_step();
            level_set_solver3.get_unp1(locally_relevant_solution_LS_3); // exposes interface vector

          }
    }

    //    if (get_output && time - (output_number)*output_time > 0)
    //      output_results();
  if(timestep_number%1==0)
    {
      display_vectors();
    }
    // output_vectors_Q();
    prepare_next_time_step();
  } // end of time loop

}

} // end namespace CPPLS

constexpr int dim{2};

int main(int argc, char* argv[])
{
  // One of the new features in C++11 is the <code>chrono</code> component of
  // the standard library. This gives us an easy way to time the output.
  try {
    using namespace dealii;
    using namespace CPPLS;

    auto t0 = std::chrono::high_resolution_clock::now();

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    CPPLS::Parameters parameters;
    parameters.read_parameter_file("parameters.prm");
    CPPLS::MaterialData material_data;
    {
    LayerMovementProblem<dim> run_layers(parameters, material_data);
    run_layers.run();
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
