

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

#include "my_utility_functions.h"
#include "physical_functions.h"
#include "material_data.h"
#include "parameters.h"
#include "level_set_solver.h"

namespace CPPLS {
using namespace dealii;

// some free functions

//template <int dim>
//void print_mesh_info(const Triangulation<dim>& triangulation, const std::string& filename)
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

//double porosity(const double pressure, const double overburden, const double initial_porosity,
//                const double compaction_coefficient)
//{
//  return (initial_porosity * std::exp(-1 * compaction_coefficient * (overburden - pressure)));
//}
//double permeability(const double porosity, const double initial_permeability, const double initial_porosity)
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

  // output stream where only mpi rank 0 output gets to stdout
  ConditionalOStream pcout;

  TimerOutput computing_timer;

  const double time_step;
  double current_time;
  double output_number;
  double final_time;

  //set timestepping scheme 1 implicit euler, 1/2 CN, 0 explicit euler
  const double theta;

  ConstraintMatrix constraints_P;
  ConstraintMatrix constraints_T;
  ConstraintMatrix constraints_LS;
  ConstraintMatrix constraints_V;

  // FE Field Solution Vectors

  LA::MPI::Vector locally_relevant_solution_LS; // ls
  LA::MPI::Vector old_locally_relevant_solution_LS;
  LA::MPI::Vector locally_relevant_solution_P;
  LA::MPI::Vector old_locally_relevant_solution_P;
  LA::MPI::Vector locally_relevant_solution_T;
  LA::MPI::Vector locally_relevant_solution_Fx;
  LA::MPI::Vector locally_relevant_solution_Fy;// speed function
  LA::MPI::Vector completely_distributed_solution_LS;
  LA::MPI::Vector completely_distributed_solution_P;
  LA::MPI::Vector completely_distributed_solution_t;
  LA::MPI::Vector completely_distributed_solution_f;

  LA::MPI::Vector rhs_P;
  LA::MPI::Vector old_rhs_P;
  LA::MPI::Vector system_rhs_P;

  // Physical Vectors

  LA::MPI::Vector overburden;
  LA::MPI::Vector old_overburden;
  LA::MPI::Vector bulkdensity;
  LA::MPI::Vector porosity;
  LA::MPI::Vector old_porosity;
  LA::MPI::Vector permeability;

  // Sparse Matrices
  LA::MPI::SparseMatrix laplace_matrix_P;
  LA::MPI::SparseMatrix mass_matrix_P;
  LA::MPI::SparseMatrix system_matrix_P;

  // for boundary conditions
  std::vector<unsigned int> boundary_values_id_LS;

  std::vector<double> boundary_values_LS;




  // Member Functions

  // create mesh
  void setup_geometry();

  // create appropriately sized vectors and matrices
  void setup_system_P();
  void setup_system_LS();
  void initial_conditions();
  void set_boundary_inlet();
  void get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
                              std::vector<double>& boundary_values_LS);

  void setup_material_configuration();
  // initialize vectors
  void setup_dofs_P();
  void setup_dofs_T();

  void assemble_matrices_P();
  void assemble_system_T();
  //for level set, this in handled in the solver class
  void forge_system_P();
  void solve_time_step_P();

  void compute_bulkdensity();
  void compute_overburden();
  void compute_porosity();
  void compute_speed_function();

  void output_vectors();
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
  , dof_handler_P(triangulation)
  , dof_handler_T(triangulation)
  , dof_handler_LS(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
  , time_step((parameters.stop_time - parameters.start_time)
              /parameters.n_time_steps)
  ,current_time {parameters.start_time}
  ,theta(parameters.theta)

        {};

// Destructor
template <int dim>
LayerMovementProblem<dim>::~LayerMovementProblem()
{
  dof_handler_P.clear();
  dof_handler_T.clear();
  dof_handler_LS.clear();
}

//
template <int dim>
void LayerMovementProblem<dim>::setup_geometry()
{
  TimerOutput::Scope t(computing_timer, "setup_geometry");
  GridGenerator::hyper_cube(triangulation, 0, parameters.box_size, true);
  // GridGenerator::subdivided_hyper_rectangle(triangulation, 0, parameters.box_size);
  triangulation.refine_global(parameters.initial_refinement_level);
  print_mesh_info(triangulation, "my_grid");
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

  //initialize
  porosity.reinit(locally_owned_dofs_P, mpi_communicator);
  old_porosity.reinit(locally_owned_dofs_P, mpi_communicator);
  porosity=0.65;


  locally_relevant_solution_Fx.reinit(locally_owned_dofs_P, locally_relevant_dofs_P, mpi_communicator);
  locally_relevant_solution_Fx = 0;
  locally_relevant_solution_Fy.reinit(locally_owned_dofs_P, locally_relevant_dofs_P, mpi_communicator);
  locally_relevant_solution_Fy = 0;


  // constraints

  constraints_P.clear();

  constraints_P.reinit(locally_relevant_dofs_P);

  DoFTools::make_hanging_node_constraints(dof_handler_P, constraints_P);
  // zero dirichlet at top
  VectorTools::interpolate_boundary_values(dof_handler_P, 1, ZeroFunction<dim>(), constraints_P);
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
void LayerMovementProblem<dim>::setup_system_LS()
{
  TimerOutput::Scope t(computing_timer, "setup_system_LS");

  dof_handler_LS.distribute_dofs(fe_LS);

  pcout << std::endl
        << "============Pressure===============" << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_LS.n_dofs() << std::endl
        << std::endl;

  locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_LS, locally_relevant_dofs_LS);

  // vector setup
  locally_relevant_solution_LS.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  locally_relevant_solution_LS = 0;

  completely_distributed_solution_LS.reinit(locally_owned_dofs_LS, mpi_communicator);

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
  // init condition for P
  completely_distributed_solution_P = 0;
  VectorTools::interpolate_boundary_values(dof_handler_P, /*top boundary*/ 1, ZeroFunction<dim>(), constraints_P);
  constraints_P.distribute(completely_distributed_solution_P);
  locally_relevant_solution_P = completely_distributed_solution_P;

  // init condition for LS
  completely_distributed_solution_LS = 0;
  VectorTools::interpolate(dof_handler_LS, Initial_LS<dim>(), completely_distributed_solution_LS);
  constraints_LS.distribute(completely_distributed_solution_LS);
  locally_relevant_solution_LS = completely_distributed_solution_LS;
}

//template <int dim>
//void LayerMovementProblem<dim>::set_boundary_inlet()
//{
//  const QGauss<dim - 1> face_quadrature_formula(1); // center of the face
//  FEFaceValues<dim> fe_face_values(fe_LS, face_quadrature_formula,
//                                   update_values | update_quadrature_points | update_normal_vectors);
//  const unsigned int n_face_q_points = face_quadrature_formula.size();
//  std::vector<double> u_value(n_face_q_points);
//  std::vector<double> v_value(n_face_q_points);

//  typename DoFHandler<dim>::active_cell_iterator cell_U = dof_handler_LS.begin_active(), endc_U = dof_handler_LS.end();
//  Tensor<1, dim> u;

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

  set_boundary_inlet();
  boundary_id = 10; // inlet
  //we define the inlet to be at the top, i.e. boundary_id=1
  VectorTools::interpolate_boundary_values(dof_handler_LS, 1, BoundaryPhi<dim>(1.0), map_boundary_values_LS);
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


  FE_Q<dim> fe(1);
  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  // FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  // Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<double> bulkdensity_at_quad(n_q_points);
  std::vector<double> porosity_at_quad(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  constraints_V.close();

  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    fe_values.reinit(cell);
    // query the local porosity - use constant for now
    const double rock_density = material_data.get_solid_density(cell->material_id());

    fe_values.get_function_values(porosity, porosity_at_quad);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      bulkdensity_at_quad[q_point] =
          porosity_at_quad[q_point] * material_data.fluid_density + (1 - porosity_at_quad[q_point]) * rock_density;
    }
    cell->get_dof_indices(local_dof_indices);
    constraints_V.distribute_local_to_global(bulkdensity_at_quad, local_dof_indices, bulkdensity);
  }
  bulkdensity.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::compute_overburden()
{
  TimerOutput::Scope t(computing_timer, "compute_overburden");

  FE_Q<dim> fe(1);
  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> bulkdensity_at_quad(n_q_points);

  Point<dim> point_for_depth;

  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

    fe_values.reinit(cell);

    // fe_values.get_function_values(interface_LS, phi_at_quad);
    fe_values.get_function_values(bulkdensity, bulkdensity_at_quad);
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      point_for_depth = fe_values.quadrature_point(q_point);
      overburden_at_quad[q_point] = /* material_data.g */ 9.81 * point_for_depth[2] * bulkdensity_at_quad[q_point];
    }
    cell->get_dof_indices(local_dof_indices);
    constraints_V.distribute_local_to_global(overburden_at_quad, local_dof_indices, overburden);
  }
  overburden.compress(VectorOperation::add);
}


template <int dim>
void LayerMovementProblem<dim>::compute_porosity()
{
  TimerOutput::Scope t(computing_timer, "compute_porosity");

FE_Q<dim> fe(1);
const QGauss<dim> quadrature_formula(3);

FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);

const unsigned int dofs_per_cell = fe.dofs_per_cell;
const unsigned int n_q_points = quadrature_formula.size();

std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

std::vector<double> pressure_at_quad(n_q_points);
std::vector<double> overburden_at_quad(n_q_points);
std::vector<double> porosity_at_quad(n_q_points);

for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

  fe_values.reinit(cell);
  const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
  const double compaction_coefficient = material_data.get_compressibility_coefficient(cell->material_id());

  // fe_values.get_function_values(interface_LS, phi_at_quad);
  fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
  fe_values.get_function_values(overburden, overburden_at_quad);
  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

    porosity_at_quad[q_point] = CPPLS::porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity, compaction_coefficient) ;
  }
  cell->get_dof_indices(local_dof_indices);
  constraints_V.distribute_local_to_global(porosity_at_quad, local_dof_indices, porosity);
}
porosity.compress(VectorOperation::add);
}



template <int dim>
void LayerMovementProblem<dim>::setup_material_configuration()
{
  TimerOutput::Scope t(computing_timer, "material_move");
  //This function sets material ids of cells based on the location of the interface, i.e. loc_rel_solution_LS

  FE_Q<dim> fe(1);
  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> LS_at_quad(n_q_points);
  //std::vector<double> bulkdensity_at_quad(n_q_points);
  //double eps= GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2);
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

   double id_sum{0};

  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    id_sum=0;
    fe_values.reinit(cell);
    fe_values.get_function_values(locally_relevant_solution_LS, LS_at_quad);
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      id_sum += LS_at_quad[q_point];
    }
    if (id_sum<0)
      {
        cell->set_material_id(1);
      }
    else
      {
        cell->set_material_id(0);
      }


    }

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

  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    cell_laplace_matrix = 0;
    cell_mass_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);
    fe_values.get_function_values(porosity, porosity_at_quad);
    fe_values.get_function_values(permeability, permeability_at_quad);
    fe_values.get_function_values(overburden, overburden_at_quad);
    fe_values.get_function_values(old_overburden, old_overburden_at_quad);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double diff_coeff_at_quad =
          (permeability_at_quad[q_point] / (porosity_at_quad[q_point] * material_data.fluid_viscosity));
      const double rhs_at_quad = (overburden[q_point] - old_overburden_at_quad[q_point]) / time_step;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          cell_laplace_matrix(i, j) += diff_coeff_at_quad * (fe_values.shape_grad(i, q_point) *
                                                             fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

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
    constraints_P.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_P, rhs_P);
    constraints_P.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_P);
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


template<int dim>
  void LayerMovementProblem<dim>::forge_system_P()
  {
    // in this function we manipulate A, M, F, resulting from assemble_matrices_P
    TimerOutput::Scope t(computing_timer, "forge_P");
    LA::MPI::Vector         tmp;
    LA::MPI::Vector         forcing_terms;


    tmp.reinit (locally_owned_dofs_P,
                locally_relevant_dofs_P, mpi_communicator);
//    forcing_terms.reinit (locally_owned_dofs,
//                          locally_relevant_dofs, mpi_communicator);
    forcing_terms.reinit (locally_owned_dofs_P, mpi_communicator);

    locally_relevant_solution_P =  old_locally_relevant_solution_P;
    mass_matrix_P.vmult(system_rhs_P, old_locally_relevant_solution_P);

    laplace_matrix_P.vmult(tmp, old_locally_relevant_solution_P);
            //  pcout << "laplace symmetric: " << laplace_matrix.is_symmetric()<<std::endl;
    system_rhs_P.add(-(1 - theta) * time_step, tmp);


            forcing_terms.add(time_step * theta, rhs_P);



            forcing_terms.add(time_step * (1 - theta), old_rhs_P);

            system_rhs_P += forcing_terms;
            //system_matrix.compress (VectorOperation::add);

            system_matrix_P.copy_from(mass_matrix_P);
            //system_matrix.compress (VectorOperation::add);

            system_matrix_P.add(laplace_matrix_P, time_step*(1-theta));

            //constraints.condense (system_matrix, system_rhs);

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



template<int dim>
  void LayerMovementProblem<dim>::solve_time_step_P()
  {
    TimerOutput::Scope t(computing_timer, "solve_time_step_P");

    LA::MPI::Vector completely_distributed_solution (locally_owned_dofs_P, mpi_communicator);


    SolverControl solver_control (dof_handler_P.n_dofs(), 1e-12);
    LA::SolverCG solver(solver_control, mpi_communicator);

    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

    data.symmetric_operator = true;
    preconditioner.initialize(system_matrix_P, data);

      solver.solve (system_matrix_P, completely_distributed_solution, system_rhs_P,
                  preconditioner);

    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;

    constraints_P.distribute (completely_distributed_solution);

    locally_relevant_solution_P = completely_distributed_solution;
  }




template <int dim>
void LayerMovementProblem<dim>::compute_speed_function()
{
  FE_Q<dim> fe(1);
  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> old_porosity_at_quad(n_q_points);
  std::vector<double> Fx_at_quad(n_q_points);
  std::vector<double> Fy_at_quad(n_q_points);

  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

    fe_values.reinit(cell);

    // fe_values.get_function_values(interface_LS, phi_at_quad);
    fe_values.get_function_values(porosity, porosity_at_quad);
    fe_values.get_function_values(old_porosity, old_porosity_at_quad);
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

      Fx_at_quad[q_point]=0;
      Fy_at_quad[q_point]=0.25;//porosity_at_quad[q_point]-old_porosity_at_quad[q_point]
    }
    cell->get_dof_indices(local_dof_indices);
    constraints_V.distribute_local_to_global(Fx_at_quad, local_dof_indices, locally_relevant_solution_Fx);
    constraints_V.distribute_local_to_global(Fy_at_quad, local_dof_indices, locally_relevant_solution_Fy);
  }
  locally_relevant_solution_Fx.compress(VectorOperation::add);
  locally_relevant_solution_Fy.compress(VectorOperation::add);

}



template <int dim>
void LayerMovementProblem<dim>::output_vectors()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_LS);
  data_out.add_data_vector(locally_relevant_solution_LS, "LS");
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

    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim>
void LayerMovementProblem<dim>::run()
{
  // common mesh
  setup_geometry();
  setup_system_P();
  setup_system_LS();
  initial_conditions();

  // at this point the p system and ls system should be ready for the time loop

  // initialize level set solver
  //we use some hardcode defaults for now
  const double cK=1.0;
  const double cE=1.0;
  const bool verbose =true;
  std::string ALGORITHM="MPP_uH";
  const unsigned int TIME_INTEGRATION=1; //corresponds to SSP33
  LevelSetSolver<dim> level_set_solver(degree_LS, degree_LS,
         time_step,
         cK,
         cE,
          verbose,
         ALGORITHM,
          TIME_INTEGRATION,
           triangulation,
         mpi_communicator);

  // initialize pressure solver
  // PressureEquation<dim> pressure_solver;
  // initialize temperature solver
  // TemperatureEquation<dim> temperature_solver;


  //  // TIME STEPPING
  for (double timestep_number = 1, time = time_step; time <= final_time; time += time_step, ++timestep_number) {
    pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

    //TODO: put the level set solve first in the time loop like Octave


    //set material ids based on locally_relevant_solution_LS
    setup_material_configuration();
    //set physical vectors
    compute_bulkdensity();
    compute_overburden();

    compute_porosity();

    //assemble and solve for pressure
    assemble_matrices_P();
    forge_system_P();
    solve_time_step_P();

    compute_speed_function();//set locally_relevant_solution_u/Fx, locally_relevant_solution_v/Fy

    // Level set computation
    //original level_set_solver.set_velocity(locally_relevant_solution_u, locally_relevant_solution_v);
    level_set_solver.set_velocity(locally_relevant_solution_Fx, locally_relevant_solution_Fy);
    level_set_solver.nth_time_step();
    level_set_solver.get_unp1(locally_relevant_solution_LS); // exposes interface vector


//    if (get_output && time - (output_number)*output_time > 0)
//      output_results();
  }
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

    LayerMovementProblem<dim> run_layers(parameters, material_data);
    run_layers.run();

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
