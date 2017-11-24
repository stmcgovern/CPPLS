#ifndef dealii__cppls_physical_functions_h
#define dealii__cppls_physical_functions_h

#include "parameters.h"

// Functions for the initial values and boundary values of pressure and level set
// pressure initial is 0 and boundary is homogeneous dirichlet at top
// level set is constant down with some initial values

namespace CPPLS {
using namespace dealii;

// Porosity and permeability are "empirical" relations that can be changed here (e.g., Kozeny-Carman or linear
// porosity,etc.)
double porosity(const double pressure, const double overburden, const double initial_porosity,
                const double compaction_coefficient, const double hydrostatic)
{
    return (initial_porosity * std::exp(-1 * compaction_coefficient * (overburden - pressure - hydrostatic)));
}
double permeability(const double porosity, const double initial_permeability, const double initial_porosity)
{
    return (initial_permeability * (1 - initial_porosity) / (1 - porosity));
}

// These functions are definitional

double bulkdensity(const double porosity, const double fluid_density, const double solid_density)
{
    return (porosity * fluid_density + (1 - porosity) * solid_density);
}
// Note that our "pressure" variable is the OVERPRESSURE, not the pore pressure
double VES(const double overburden, const double pressure, const double hydrostatic)
{
    return (overburden - pressure - hydrostatic);
}

double bulkheatcapacity(const double porosity, const double fluid_heat_capacity, const double solid_heat_capacity)
{
    return (porosity * fluid_heat_capacity + (1 - porosity) * solid_heat_capacity);
}



//template <int dim>
//class LayerMovementProblem<dim>::Postprocessor : public DataPostprocessor<dim>
//{
//public:
//  Postprocessor (const CPPLS::MaterialData& material_data,
//                 const CPPLS::Parameters& parameters);
//  virtual
//  void
//  evaluate_vector_field
//  (const DataPostprocessorInputs::Vector<dim> &inputs,
//   std::vector<Vector<double> >               &computed_quantities) const override;
//  virtual std::vector<std::string> get_names () const override;
//  virtual
//  std::vector<DataComponentInterpretation::DataComponentInterpretation>
//  get_data_component_interpretation () const override;
//  virtual UpdateFlags get_needed_update_flags () const override;
//private:
//  const CPPLS::MaterialData& material_data;
//  const CPPLS::Parameters& parameters;
//};
//template <int dim>
//LayerMovementProblem<dim>::Postprocessor::
//Postprocessor (const CPPLS::MaterialData& material_data,
//               const CPPLS::Parameters& parameters)
//  :
//  material_data (material_data),
//  parameters (parameters)
//{}
//template <int dim>
//std::vector<std::string>
//LayerMovementProblem<dim>::Postprocessor::get_names() const
//


template <int dim>
class SedimentationRate : public Function<dim> {
public:
    SedimentationRate(double t, const CPPLS::Parameters& parameters)
        : Function<dim>(),
          parameters(parameters){}
//    {
//        this->set_time(t);
//    }
    virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;

    virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                            const unsigned int component) const override;
private:
    const CPPLS::Parameters& parameters;

};
// Sedimentation rate in the <dim 2> along x and <dim 3> x-y plane
template <int dim>
double SedimentationRate<dim>::value(const Point<dim>& p, const unsigned int) const
{

    double return_value = 10;
    const double time = this->get_time();
    const double magnify=0.3;

    switch (dim) {
    case 1: {
        Assert(false, ExcNotImplemented());
        break;
    }
    case 2: {
        double x = p[0];
        double y = p[1];
        double mid =x -parameters.box_size/2;
        double left=x -parameters.box_size/5;
        double right=x -3*parameters.box_size/4;

        if(time<(parameters.stop_time/3))
        {
           return_value=-1*parameters.base_sedimentation_rate*(1+1.1*magnify*sin(numbers::PI*x/1000));
        }
        else if(time<(2*parameters.stop_time/3))
          {
          return_value=-1*parameters.base_sedimentation_rate*(1+magnify*std::exp(-1*(mid)*(mid)/(2*parameters.box_size))
                                                              +0.1*magnify*sin(numbers::PI*x/1000));
           }
        else
          {
          return_value=-1*parameters.base_sedimentation_rate*(1-magnify*std::exp(-1*(left)*(left)/(parameters.box_size))
                                                              -(2*magnify)*std::exp(-1*(right)*(right)/(parameters.box_size)));
           }


        //return std::abs(sin(x)); return (-3.15e-11*(1+0.1*std::abs(sin(x))));
        //return (-3.15e-11*(1+0.2*std::abs(sin(numbers::PI*x/200))));
        //return (-3.15e-11*(1+0.1*sin(numbers::PI*x/200)));
        //return (-3.15e-11*(1+(x/10000)));

        break;

    }
    case 3: {
        double x = p[0];
        double y = p[1];
        double z = p[2];

        return_value=-1*parameters.base_sedimentation_rate;
        break;

    }
    default:
        Assert(false, ExcNotImplemented());
        break;
    }
    return return_value;
}

template <int dim>
void SedimentationRate<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                                        const unsigned int component) const
{
    Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));

    for (unsigned int i = 0; i < points.size(); ++i)
        values[i] = SedimentationRate<dim>::value(points[i], component);
}

// For now this is meant to implement the vertical direction unit vector for the SUPG parts
template <int dim>
class AdvectionField : public TensorFunction<1, dim> {
public:
    AdvectionField()
        : TensorFunction<1, dim>()
    {
    }
    virtual Tensor<1, dim> value(const Point<dim>& p) const;
    virtual void value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& values) const;
    DeclException2(ExcDimensionMismatch, unsigned int, unsigned int,
                   << "The vector has size " << arg1 << " but should have " << arg2 << " elements.");
};
template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim>& p) const
{

    Point<dim> value;

    switch (dim) {
    case 1: {
        Assert(false, ExcNotImplemented());
        break;
    }
    case 2: {
          value[0] = 0;
          value[1] = -1;
          break;

        }
    case 3:{
          value[0] = 0;
          value[1] = 0;
          value[2] = -1;
         break;


        }
      default:
          Assert(false, ExcNotImplemented());
        break;
      }
    return value;

}
template <int dim>
void AdvectionField<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& values) const
{
    Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
    for (unsigned int i = 0; i < points.size(); ++i)
        values[i] = AdvectionField<dim>::value(points[i]);
}

/////////////////////////////////////////////////////
//////////////////// INITIAL Level Set ////////////////////
/////////////////////////////////////////////////////
template <int dim>
class Initial_LS : public Function<dim> {
public:
    Initial_LS(double sharpness = 0.0005, const double box_size_z = 1)
        : Function<dim>()
        , sharpness(sharpness)
        , box_size_z(box_size_z)
    {
    }
    virtual double value(const Point<dim>& p, const unsigned int component = 0) const;
    double sharpness;
    const double box_size_z;
};
template <int dim>
double Initial_LS<dim>::value(const Point<dim>& p, const unsigned int) const
{
    double return_value;
    switch (dim) {
    case 1: {
        Assert(false, ExcNotImplemented());
        break;
    }
    case 2: {

        double x = p[0];
        double y = p[1];
        // std::cout<<std::tanh((y- 0.9)/sharpness)<<" "<<std::endl;

        return_value = 0.5*(1+ std::tanh((y - (box_size_z - (box_size_z / 100))) / sharpness));
        Assert (return_value <= 1 ,ExcInternalError());
        Assert (return_value >= -1 ,ExcInternalError());

        // return std::tanh((y-0.99)/sharpness);
        break;
    }
    case 3: {
        double x = p[0];
        double y = p[1];
        double z = p[2];
//        return 0.5 * (-std::tanh((y - 0.3) / sharpness) * std::tanh((y - 0.35) / sharpness) + 1) *
//               (-std::tanh((x - 0.02) / sharpness) + 1) -
//               1;
        return_value = 0.5*(1+ std::tanh((z - (box_size_z - (box_size_z / 100))) / sharpness));
        Assert (return_value <= 1 ,ExcInternalError());
        Assert (return_value >= -1 ,ExcInternalError());

        break;
    }
    default:
        Assert(false, ExcNotImplemented());
        break;
    }
     return return_value;
}




/////////////////////////////////////////////////////
//////////////////// BOUNDARY PHI ///////////////////
/////////////////////////////////////////////////////
template <int dim>
class BoundaryPhi : public ConstantFunction<dim> {
public:
    BoundaryPhi(const double value, const unsigned int n_components = 1)
        : ConstantFunction<dim>(value, n_components)
    {
    }
};


} // namespace CPPLS

#endif
