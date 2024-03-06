#include <math.h>
#include <nlopt.h>
#include <stdio.h>



double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    if (grad) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}

typedef struct {
    double a, b;
} my_constraint_data;


// typedef struct {
//     double ceq_1;
//     double ceq_2;
// }myequalityconstraints_data;

// typedef struct {
//     double cin_1;
// }myinequalityconstraints_data;


double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
    my_constraint_data *d = (my_constraint_data *) data;
    double a = d->a, b = d->b;
    if (grad) {
        grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
        grad[1] = -1.0;
    }
    return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
 }


// double myequalityconstraints(unsigned m, double *result, unsigned n,
//                              const double *x,  double *grad,
//                              void *equalitydata)
// {
//     myequalityconstraints_data *data = (myequalityconstraints_data *) equalitydata;

//     double c1 = data->ceq_1;
//     double c2 = data->ceq_2;
//     double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4];
//     result[0] = x1+x2+x3-c1; //5;
//     result[1] = x3*x3+x4-c2; //2;
//  }

//  double myinequalityconstraints(unsigned m, double *result, unsigned n,
//                                 const double *x,  double *grad,
//                                 void* inequalitydata)
//  {
//      myinequalityconstraints_data *data = (myinequalityconstraints_data *) inequalitydata;

//      double c1 = data->cin_1;
//      double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4];
//      result[0] = x4*x4+x5*x5-c1; //5;
//   }


int main(int argc, char *argv[])
{

// //establish sizes
// unsigned n = 5; //number of decision variables
// unsigned m_eq = 2; //number of equality constraints
// unsigned m_in = 1; //number of inequality constraints


double lb[2] = { -HUGE_VAL, 0 }; /* lower bounds */
nlopt_opt opt;

opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
nlopt_set_lower_bounds(opt, lb);
nlopt_set_min_objective(opt, myfunc, NULL);

my_constraint_data data[2] = { {2,0}, {-1,1} };

nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);
nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-8);

nlopt_set_xtol_rel(opt, 1e-4);


double x[2] = { 1.234, 5.678 };  /* `*`some` `initial` `guess`*` */
double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
if (nlopt_optimize(opt, x, &minf) < 0) {
    printf("nlopt failed!\n");
}
else {
    printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
}

nlopt_destroy(opt);


}