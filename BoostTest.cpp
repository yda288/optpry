#define BOOST_PYTHON_STATIC_LIB
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

namespace np = boost::python::numpy;
namespace python = boost::python;

//European call pricing only
double montecarlo(double _S0, double _K, double _r, double _div, double _T, double _sigma)
{

	static boost::mt19937_64 igen;
	static boost::variate_generator<boost::mt19937_64, boost::normal_distribution<> >
	gen(igen,
			boost::normal_distribution<>());

	double sum = 0.;
	int N = 1e5;
	for (int i = 0; i<N; i++)
	{
		double phi = gen();
		double ST = _S0 * exp((_r - _div - 0.5*_sigma*_sigma)*_T + phi*_sigma*sqrt(_T));
		sum = sum + std::max(ST - _K, 0.);
	}
	return sum / N*exp(-_r*_T);
}

 np::ndarray mc_harness(const boost::python::object& object) {

	np::dtype dt = np::dtype::get_builtin<double>();
	python::tuple shape = python::make_tuple(1000);
	np::ndarray trials = np::zeros(shape, dt);

	boost::python::extract<float> _S0 = object.attr("_S0");
	boost::python::extract<float> _K = object.attr("_K");
	boost::python::extract<float> _r = object.attr("_r");
	boost::python::extract<float> _div = object.attr("_div");
	boost::python::extract<float> _T = object.attr("_T");
	boost::python::extract<float> _sigma = object.attr("_sigma");

	for (int i = 0; i<1e3; ++i)
		trials[i] = montecarlo(_S0, _K, _r, _div, _T, _sigma);

	return trials;
}



BOOST_PYTHON_MODULE(BoostTest)
{
	
	Py_Initialize();
	np::initialize();

	python::def("montecarlo", &mc_harness);

}


