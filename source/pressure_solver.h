
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
}





#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/tria.h>
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
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>


#include <fstream>
#include <iostream>


namespace Pcls1
{
  using namespace dealii;



  template<int dim>
  class PressureEquation
  {
  public:
    PressureEquation();
    void initialize();
    void solve_time_step();
    

  private:
    void setup_system();
  
    void output_results() const;
    void output_results2();
    //void refine_mesh (const unsigned int min_grid_level,
                      const unsigned int max_grid_level);
    void process_solution(const double cycle);

    MPI_Comm             &mpi_communicator;
    parallel::distributed::Triangulation<dim>   &triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    IndexSet             locally_owned_dofs;
    IndexSet             locally_relevant_dofs;


    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    LA::MPI::SparseMatrix mass_matrix;
    LA::MPI::SparseMatrix laplace_matrix;
    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector       solution;
    LA::MPI::Vector       old_solution;
    LA::MPI::Vector       system_rhs;

    LA::MPI::Vector		 VES;
    LA::MPI::Vector		 overburden;
    LA::MPI::Vector		 porosity;
    LA::MPI::Vector	     density;//of rock

    LA::MPI::Vector       level_set;

    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;

    double               time;
    double               time_step;
    unsigned int         timestep_number;

    const double         theta;
    ConvergenceTable     convergence_table;

    const double		 k_perm =std::pow(10,-18);
    const double		 mu=0.001;
    const double		 rhof=1024;
    const double		 rhos=2720;
    const double		 alph=std::pow(10,-8);
    const double		 g=9.81;
    const double		 wb=1000;//3.1746*std::pow(10,-11);

  };





  template <int dim>
  class Coefficient : public Function<dim>
  {
  public:
    Coefficient ()  : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

  template <int dim>
  double Coefficient<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
  {
	  const double time = this->get_time();

        const double		 k_perm =std::pow(10,-18);
	    const double		 mu=0.001;
	    const double		 rhof=1024;
	    const double		 rhos=2720;
	    const double		 alph=std::pow(10,-8);
	    const double		 g=9.81;
	    const double		 wb=1000;//3.1746*std::pow(10,-11);

	    double return_value=k_perm*60*60*24*365*1000000/(alph*mu);//std::pow(10,-7);// k_perm/(alph*mu);

	    double increment= wb*time;

		  if (p[1]>(1000-increment))
		  { //std::cout<<"condition met"<<std::endl;
			  return return_value;//return_value;
		  }
		  else
		  {
			  return std::pow(10,-12);
		  }
  }


template <int dim>
class ExactSolution: public Function<dim>
{
public:
	ExactSolution () : Function<dim>() {}
	virtual double value (const Point<dim>   &p,
	                            const unsigned int  component = 0) const;
};
template <int dim>
  double ExactSolution<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
  {
	    const double time=1;
	    const double		 k_perm =std::pow(10,-18);
	    const double		 mu=0.001;
	    const double		 rhof=1024;
	    const double		 rhos=2720;
	    const double		 rhob=2100;//at 0.61 porosity
	    const double		 alph=std::pow(10,-8);
	    const double		 g=9.81;
	    const double		 wb=1000;//3.1746*std::pow(10,-11);
		double c=k_perm*60*60*24*365*1000000/(alph*mu);
	    double gamma = (rhob-rhof)*g;
		double expression;
		double integral =0;
		const double dx=0.1;
		for (double x=0;x<1000; x+=dx)
		{
			integral += x*std::tanh(wb*x/(2*c))*std::cosh((1000-p[1])*x/(2*c*time))*std::exp(-x*x/(4*c*time)) *dx;
		}
		expression= gamma*wb*time - gamma*(1/std::sqrt(numbers::PI * c*time))*std::exp(-(1000-p[1])*(1000-p[1])/(4*c*time))*integral;
		return expression;
}

  template <int dim>
    class Speed_function : public Function<dim>
    {
    public:
      Speed_function ()  : Function<dim>() {}
      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      virtual void value_list (const std::vector<Point<dim> > &points,
                               std::vector<double>            &values,
                               const unsigned int              component = 0) const;
    };

    template <int dim>
    double Speed_function<dim>::value (const Point<dim> &p,
                                    const unsigned int /*component*/) const
    {
    	return 0.5;
    }

    template <int dim>
    void Speed_function<dim>::value_list (const std::vector<Point<dim> > &points,
                                       std::vector<double>            &values,
                                       const unsigned int              component) const
    {
      Assert (values.size() == points.size(),
              ExcDimensionMismatch (values.size(), points.size()));
      Assert (component == 0,
              ExcIndexRange (component, 0, 1));
      const unsigned int n_points = points.size();
    }






  template<int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide ()
      :
      Function<dim>(),
      period (0.2)
    {}

    virtual double value (const Point<dim> &p,
        const unsigned int component = 0) const;

  private:
    const double period;
  };



  template<int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
            const unsigned int component) const
  {
	  const double time = this->get_time();
	  const double		 k_perm =std::pow(10,-18);
	  	    const double		 mu=0.001;
	  	    const double		 rhof=1024;
	  	    const double		 rhos=2720;
	  	    const double		 rhob=2100;//at 0.61 porosity
	  	    const double		 alph=std::pow(10,-8);
	  	    const double		 g=9.81;
	  	    const double		 wb=1000; //3.1746*std::pow(10,-11);
    Assert (component == 0, ExcInternalError());
    Assert (dim == 2, ExcNotImplemented());
    double return_value;

    double increment= wb*time;
    return_value=rhob*g*wb;



    		  if (p[1]>(1000-increment))
    		  { //std::cout<<"condition met"<<std::endl;
    			  return return_value;//return_value;
    		  }
    		  else
    		  {
    			  return 0;
    		  }

  }



  template<int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value (const Point<dim>  &p,
        const unsigned int component = 0) const;
  };


  
  template<int dim>
  double BoundaryValues<dim>::value (const Point<dim> &p,
             const unsigned int component) const
  {
    Assert(component == 0, ExcInternalError());
    if (p[1] == 1000)
    	return 0;
    else
     	return 0;
  }

  template<int dim>
   class InitialValues : public Function<dim>
   {
   public:
     virtual double value (const Point<dim>  &p,
         const unsigned int component = 0) const;
   };



   template<int dim>
   double InitialValues<dim>::value (const Point<dim> &p,
              const unsigned int component) const
   {
     Assert(component == 0, ExcInternalError());
     if (p[1] == 1)
     	return 40;
     else
      	return 0;
   }



template<int dim>
    PressureEquation<dim>::PressureEquation ()
    :
	mpi_communicator (MPI_COMM_WORLD),
	triangulation (mpi_communicator,
	                   typename Triangulation<dim>::MeshSmoothing
	                   (Triangulation<dim>::smoothing_on_refinement |
	                    Triangulation<dim>::smoothing_on_coarsening)),
    fe(1),
    dof_handler(triangulation),
    time_step(1. /50),
    theta(1),
	pcout (std::cout,
	           (Utilities::MPI::this_mpi_process(mpi_communicator)
	            == 0)),
	    computing_timer (mpi_communicator,
	                     pcout,
	                     TimerOutput::summary,
	                     TimerOutput::wall_times)
  {}



  template<int dim>
  void PressureEquation<dim>::setup_system()
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

	    typename DoFHandler<dim>::active_cell_iterator
	    cell = dof_handler.begin_active(),
	    endc = dof_handler.end();
	    for (; cell!=endc; ++cell)
	      if (cell->is_locally_owned())
	        {
	          cell_matrix = 0;
	          cell_rhs = 0;

	          fe_values.reinit (cell);

	          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	            {
	              const double
	              rhs_value
	                = (fe_values.quadrature_point(q_point)[1]
	                   >
	                   0.5+0.25*std::sin(4.0 * numbers::PI *
	                                     fe_values.quadrature_point(q_point)[0])
	                   ? 1 : -1);

	              for (unsigned int i=0; i<dofs_per_cell; ++i)
	                {
	                  for (unsigned int j=0; j<dofs_per_cell; ++j)
	                    cell_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
	                                         fe_values.shape_grad(j,q_point) *
	                                         fe_values.JxW(q_point));

	                  cell_rhs(i) += (rhs_value *
	                                  fe_values.shape_value(i,q_point) *
	                                  fe_values.JxW(q_point));
	                }
	            }

	          cell->get_dof_indices (local_dof_indices);
	          constraints.distribute_local_to_global (cell_matrix,
	                                                  cell_rhs,
	                                                  local_dof_indices,
	                                                  system_matrix,
	                                                  system_rhs);
	        }

	    system_matrix.compress (VectorOperation::add);
	    system_rhs.compress (VectorOperation::add);




	TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "==========================================="
              << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs ();

    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                                 locally_relevant_dofs);
    locally_relevant_solution.reinit (locally_owned_dofs,
                                          locally_relevant_dofs, mpi_communicator);


    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    constraints.close();

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(c_sparsity);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(locally_owned_dofs,
            locally_owned_dofs,
            dsp,
            mpi_communicator);

    Coefficient<dim> coefficient;
    coefficient.set_time(time);
    Coefficient<dim> * coeffp;
    coeffp=&coefficient;
    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      mass_matrix,
                                      (const Function<dim> *)0,
                                      constraints);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree+1),
                                         laplace_matrix,
                                         (const Function<dim> *)coeffp,
                                         constraints);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }
/*
  template<int dim>
    void HeatEquation<dim>::reinitialize_level_set()
    {
	  level_set.reinit(sparsity_pattern);
	  while (time <= 0.2)
	        {
	          time += time_step;
	          ++timestep_number;
	        }

    }
*/


  template<int dim>
  void PressureEquation<dim>::solve_time_step()
  {
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs,
             preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }


  template <int dim>
    void PressureEquation<dim>::process_solution (const double cycle)
    {
      Vector<float> difference_per_cell (triangulation.n_active_cells());
      VectorTools::integrate_difference (dof_handler,
                                         solution,
                                         ExactSolution<dim>(),
                                         difference_per_cell,
                                         QGauss<dim>(3),
                                         VectorTools::L2_norm);
      const double L2_error = difference_per_cell.l2_norm();



      const unsigned int n_active_cells=triangulation.n_active_cells();
      const unsigned int n_dofs=dof_handler.n_dofs();

      std::cout << "Cycle " << cycle << ':'
                << std::endl
                << "   Number of active cells:       "
                << n_active_cells
                << std::endl
                << "   Number of degrees of freedom: "
                << n_dofs
                << std::endl;

      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", n_active_cells);
      convergence_table.add_value("dofs", n_dofs);
      convergence_table.add_value("L2", L2_error);
    }

  template<int dim>
  void PressureEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    const std::string filename = "solution-"
                                 + Utilities::int_to_string(timestep_number, 3) +
                                 ".vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }

  template<int dim>
   void PressureEquation<dim>::output_results2()
   {
	  enum RefinementMode
	      {
	        global_refinement, adaptive_refinement
	      };
     RefinementMode refinement_mode;
	  refinement_mode=global_refinement;
     std::string vtk_filename;
         switch (refinement_mode)
           {
           case global_refinement:
             vtk_filename = "solution-global";
             break;
           case adaptive_refinement:
             vtk_filename = "solution-adaptive";
             break;
           default:
             Assert (false, ExcNotImplemented());
           }


             vtk_filename += "-q1";


         vtk_filename += ".vtk";
         std::ofstream output2 (vtk_filename.c_str());

         DataOut<dim> data_out;
         data_out.attach_dof_handler (dof_handler);
         data_out.add_data_vector (solution, "solution");


         data_out.build_patches ();
         data_out.write_vtk (output2);

         convergence_table.set_precision("L2", 3);
         convergence_table.set_precision("cycle", 3);

         convergence_table.set_scientific("L2", true);

         convergence_table.set_tex_caption("cells", "\\# cells");
         convergence_table.set_tex_caption("dofs", "\\# dofs");
         convergence_table.set_tex_caption("L2", "$L^2$-error");

         convergence_table.set_tex_format("cells", "r");
         convergence_table.set_tex_format("dofs", "r");

         std::cout << std::endl;
         convergence_table.write_text(std::cout);

         std::string error_filename = "error";
         switch (refinement_mode)
           {
           case global_refinement:
             error_filename += "-global";
             break;
           case adaptive_refinement:
             error_filename += "-adaptive";
             break;
           default:
             Assert (false, ExcNotImplemented());
           }


             error_filename += "-q1";


         error_filename += ".tex";
         std::ofstream error_table_file(error_filename.c_str());

         convergence_table.write_tex(error_table_file);
         refinement_mode=adaptive_refinement;
         if (refinement_mode==global_refinement)
           {
             convergence_table.add_column_to_supercolumn("cycle", "n cells");
             convergence_table.add_column_to_supercolumn("cells", "n cells");

             std::vector<std::string> new_order;
             new_order.push_back("n cells");
             new_order.push_back("L2");
             convergence_table.set_column_order (new_order);

             convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
             convergence_table
             .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);

             std::cout << std::endl;
             convergence_table.write_text(std::cout);

             std::string conv_filename = "convergence";
             switch (refinement_mode)
               {
               case global_refinement:
                 conv_filename += "-global";
                 break;
               case adaptive_refinement:
                 conv_filename += "-adaptive";
                 break;
               default:
                 Assert (false, ExcNotImplemented());
               }

                 conv_filename += "-q1";

             conv_filename += ".tex";

             std::ofstream table_file(conv_filename.c_str());
             convergence_table.write_tex(table_file);
           }
   }


  template <int dim>
  void PressureEquation<dim>::refine_mesh (const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(fe.degree+1),
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                                                       estimated_error_per_cell,
                                                       0.6, 0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag ();
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level); ++cell)
      cell->clear_coarsen_flag ();


    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement ();
    setup_system ();

    solution_trans.interpolate(previous_solution, solution);
  }




  template<int dim>
  void PressureEquation<dim>::run()
  {
    const unsigned int initial_global_refinement = 6;
    const unsigned int n_adaptive_pre_refinement_steps = 0;//4;

    GridGenerator::hyper_cube (triangulation, 0, 1000);

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary()
            &&
            (cell->face(f)->center()[dim-1] == 1000))
          cell->face(f)->set_boundary_indicator(1);

    triangulation.refine_global (initial_global_refinement);
    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp;
    Vector<double> forcing_terms;


    tmp.reinit (solution.size());
    forcing_terms.reinit (solution.size());

    VectorTools::interpolate(dof_handler,
                             ZeroFunction<dim>(),
                             old_solution);
    solution = old_solution;

    timestep_number = 0;
    time            = 0;

    output_results();

    while (time <= 1)
      {
        time += time_step;
        ++timestep_number;

        tmp.reinit (solution.size());
        forcing_terms.reinit (solution.size());
        setup_system();


        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution);

        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step, tmp);

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree+1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= time_step * theta;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree+1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add(time_step * (1 - theta), tmp);

        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step, laplace_matrix);

        constraints.condense (system_matrix, system_rhs);

        {
          BoundaryValues<dim> boundary_values_function;
          boundary_values_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   1,
                                                   ZeroFunction<dim>(),
												   boundary_values);

          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }

        solve_time_step();




        output_results();


        old_solution = solution;
      }//end of while loop


    process_solution(time);
    output_results2();
  }
}

