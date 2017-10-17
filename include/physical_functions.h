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

template <int dim>
class SedimentationRate : public Function<dim> {
public:
  SedimentationRate(double t = 0)
    : Function<dim>()
  {
    this->set_time(t);
  }
  virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component) const;
};
// Sedimentation rate in the <dim 2> along x and <dim 3> x-y plane
template <int dim>
double SedimentationRate<dim>::value(const Point<dim>& p, const unsigned int) const
{

  double return_value = 0;

  switch (dim) {
    case 1: {
      Assert(false, ExcNotImplemented());
      break;
    }
    case 2: {
      double x = p[0];
      double y = p[1];

      return std::abs(sin(x));

      // return return_value;

      break;
    }
    case 3: {
      double x = p[0];
      double y = p[1];
      double z = p[2];
      if (z == 1) {
        return_value = 0.25;
      }
      return return_value;
      break;
    }
    default:
      Assert(false, ExcNotImplemented());
  }
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
  value[0] = 0;
  value[1] = -1;
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

  switch (dim) {
    case 1: {
      Assert(false, ExcNotImplemented());
      break;
    }
    case 2: {

      double x = p[0];
      double y = p[1];
      // std::cout<<std::tanh((y- 0.9)/sharpness)<<" "<<std::endl;
      return std::tanh((y - (box_size_z - (box_size_z / 100))) / sharpness);
      // return std::tanh((y-0.99)/sharpness);
      break;
    }
    case 3: {
      double x = p[0];
      double y = p[1];
      double z = p[2];
      return 0.5 * (-std::tanh((y - 0.3) / sharpness) * std::tanh((y - 0.35) / sharpness) + 1) *
                 (-std::tanh((x - 0.02) / sharpness) + 1) -
             1;
      break;
    }
    default:
      Assert(false, ExcNotImplemented());
  }

  //  double x = p[0]; double y = p[1];
  //  double pi=numbers::PI;

  ////  if (PROBLEM==FILLING_TANK)
  ////    return 0.5*(-std::tanh((y-0.3)/sharpness)*std::tanh((y-0.35)/sharpness)+1)
  ////      *(-std::tanh((x-0.02)/sharpness)+1)-1;
  ////  else if (PROBLEM==BREAKING_DAM)
  ////    return 0.5*(-std::tanh((x-0.35)/sharpness)*std::tanh((x-0.65)/sharpness)+1)
  ////      *(1-std::tanh((y-0.35)/sharpness))-1;
  ////  else if (PROBLEM==FALLING_DROP)
  ////    {
  ////      double x0=0.15; double y0=0.75;
  ////      double r0=0.1;
  ////      double r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2));
  ////      return 1-(std::tanh((r-r0)/sharpness)+std::tanh((y-0.3)/sharpness));
  ////    }
  ////  else if (PROBLEM==SMALL_WAVE_PERTURBATION)
  ////    {
  ////      double wave = 0.1*std::sin(pi*x)+0.25;
  ////      return -std::tanh((y-wave)/sharpness);
  ////    }
  ////  else
  ////    {
  ////      std::cout << "Error in type of PROBLEM" << std::endl;
  ////      abort();
  ////    }
}

///////////////////////////////////////////////////////
//////////////////// FORCE TERMS ///// ////////////////
///////////////////////////////////////////////////////
template <int dim>
class ForceTerms : public ConstantFunction<dim> {
public:
  ForceTerms(const std::vector<double> values)
    : ConstantFunction<dim>(values)
  {
  }
};

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

//////////////////////////////////////////////////////////
//////////////////// BOUNDARY Values Level Set ////////////
//////////////////////////////////////////////////////////
// template <int dim>
// class BoundaryU : public Function <dim>
//{
// public:
//  BoundaryU (double t=0) : Function<dim>() {this->set_time(t);}
//  virtual double value (const Point<dim> &p, const unsigned int component=0) const;

//};
// template <int dim>
// double BoundaryU<dim>::value (const Point<dim> &p, const unsigned int) const
//{
//  //////////////////////
//  // FILLING THE TANK //
//  //////////////////////
//  // boundary for filling the tank (inlet)

//    double return_value=0;

//  switch(dim)
//    {
//    case 1:{
//      Assert(false, ExcNotImplemented());
//      break;}
//    case 2:{
//      double x=p[0]; double y=p[1];
//      if (y==1){
//        return_value = 0.25;
//      }
//      return return_value;

//      break;}
//    case 3:{
//      double x=p[0]; double y=p[1]; double z=p[2];
//      if (z==1){
//        return_value = 0.25;
//      }
//       return return_value;
//      break;}
//    default:
//       Assert(false, ExcNotImplemented());
//    }

//}

// template <int dim>
// class BoundaryV : public Function <dim>
//{
// public:
//  BoundaryV (unsigned int PROBLEM, double t=0) : Function<dim>(), PROBLEM(PROBLEM) {this->set_time(t);}
//  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
//  unsigned int PROBLEM;
//};
// template <int dim>
// double BoundaryV<dim>::value (const Point<dim> &p, const unsigned int) const
//{
//  // boundary for filling the tank (outlet)
//  double x = p[0]; double y = p[1];
//  double return_value = 0;

//  if (PROBLEM==FILLING_TANK)
//    {
//      if (y==0.4 && x>=0.3 && x<=0.35)
//        return_value = 0.25;
//    }
//  return return_value;
//}

} // namespace CPPLS

#endif
