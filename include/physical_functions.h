#ifndef dealii__cppls_physical_functions_h
#define dealii__cppls_physical_functions_h

#include "parameters.h"

//Functions for the initial values and boundary values of pressure and level set
//pressure initial is 0 and boundary is homogeneous dirichlet at top
//level set is constant down wiht some initial values


namespace CPPLS {
using namespace dealii;



double porosity(const double pressure, const double overburden, const double initial_porosity,
                const double compaction_coefficient)
{
  return (initial_porosity * std::exp(-1 * compaction_coefficient * (overburden - pressure)));
}
double permeability(const double porosity, const double initial_permeability, const double initial_porosity)
{
  return (initial_permeability * (1 - initial_porosity) / (1 - porosity));
}





/////////////////////////////////////////////////////
//////////////////// INITIAL Level Set ////////////////////
/////////////////////////////////////////////////////
template <int dim>
class Initial_LS : public Function <dim>
{
public:
  Initial_LS ( double sharpness=0.0005) : Function<dim>(),
                                                              sharpness(sharpness) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  double sharpness;
  unsigned int PROBLEM;
};
template <int dim>
double Initial_LS<dim>::value (const Point<dim> &p,
                               const unsigned int) const
{

  switch(dim)
    {
    case 1:{
      Assert(false, ExcNotImplemented());
      break;}
    case 2:{
        //TODO undo the hardcoded "box size"
      double x=p[0]; double y=p[1];
      std::cout<<std::tanh((y- 0.9)/sharpness)<<" "<<std::endl;
      return std::tanh((y-0.9)/sharpness);
      break;}
    case 3:{
      double x=p[0]; double y=p[1]; double z=p[2];
      return 0.5*(-std::tanh((y-0.3)/sharpness)*std::tanh((y-0.35)/sharpness)+1) *(-std::tanh((x-0.02)/sharpness)+1)-1;
      break;}
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
class ForceTerms : public ConstantFunction <dim>
{
public:
  ForceTerms (const std::vector<double> values) : ConstantFunction<dim>(values) {}
};

/////////////////////////////////////////////////////
//////////////////// BOUNDARY PHI ///////////////////
/////////////////////////////////////////////////////
template <int dim>
class BoundaryPhi : public ConstantFunction <dim>
{
public:
  BoundaryPhi (const double value, const unsigned int n_components=1) : ConstantFunction<dim>(value,n_components) {}
};

//////////////////////////////////////////////////////////
//////////////////// BOUNDARY Values Level Set ////////////
//////////////////////////////////////////////////////////
template <int dim>
class BoundaryU : public Function <dim>
{
public:
  BoundaryU (double t=0) : Function<dim>() {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;

};
template <int dim>
double BoundaryU<dim>::value (const Point<dim> &p, const unsigned int) const
{
  //////////////////////
  // FILLING THE TANK //
  //////////////////////
  // boundary for filling the tank (inlet)

    double return_value=0;

  switch(dim)
    {
    case 1:{
      Assert(false, ExcNotImplemented());
      break;}
    case 2:{
      double x=p[0]; double y=p[1];
      if (y==1){
        return_value = 0.25;
      }
      return return_value;

      break;}
    case 3:{
      double x=p[0]; double y=p[1]; double z=p[2];
      if (z==1){
        return_value = 0.25;
      }
       return return_value;
      break;}
    default:
       Assert(false, ExcNotImplemented());
    }

}

//template <int dim>
//class BoundaryV : public Function <dim>
//{
//public:
//  BoundaryV (unsigned int PROBLEM, double t=0) : Function<dim>(), PROBLEM(PROBLEM) {this->set_time(t);}
//  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
//  unsigned int PROBLEM;
//};
//template <int dim>
//double BoundaryV<dim>::value (const Point<dim> &p, const unsigned int) const
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
