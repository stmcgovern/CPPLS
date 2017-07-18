namespace my_lamda_functions
{

template<int dim>
  auto my_rhs =[]( const Point<dim> p) -> double
      {
        return p[0]+p[1];
        /*std::exp(-8*t)*std::exp(-40*Utilities::fixed_power<6>(p[0] - 1.5))
          *std::exp(-40*Utilities::fixed_power<6>(p[1]));*/
      };

//whole vector
template<int dim>
  auto porosity =[Beta]( const LA::MPI::Vector VES) -> LA::MPI::Vector
      {
        LA::MPI::Vector temp;
        temp.reinit(VES.size());
        for( unsigned int i=0; i<VES.size(); ++i)
        {
          temp[i]=std::exp(-Beta*VES[i]);
        }
        
        return temp;
      };



//OCTAVE IMPLEMENTATION FOR SPEED FUNCTION
// calculate F 
//   {   F(1)=omega;
//       for i=2:n
//           base_rate=(1/(1-porosity(i)))*rockSedRate;
//           F(i)=base_rate;
//           for m=i:-1:1
//               F(i)=F(i)+(1/(1-porosity(m))*rockSedRate/res)*((dpor(m)/(1-porosity(m))));
//           end
//       end
//     }
//DEALII
// we lose the guarantee that the m loop above really just picks out values above the point i

//whole vector //refence capture might be risky - check lifetimes
template<int dim>
  auto speed_function =[&]( const LA::MPI::Vector porosity,
                            const LA::MPI::Vector old_porosity) -> LA::MPI::Vector
      {
        LA::MPI::Vector temp;
        temp.reinit(porosity.size());
        for( unsigned int i=0; i<porosity.size(); ++i)
        {
          temp[i]=parameters.sedRate 

        }
        
        return temp;
      };


}



//see step-7 for another way to implement the gaussian, possible roundoff problems
template<int dim>
auto delta_approx = [grid_width] (const double eval_x, const double eval_y, const Point<dim> p) -> double
{
  const double x=p[0];
  const double y=p[1];
  return (std::sqrt(1/(2*numbers::PI*grid_width*grid_width))*
         std::exp(-1*(x-eval_x)*(x-eval_x)*(y-eval_y)*(y-eval_y)/ (2*grid_width*grid_width));

};


