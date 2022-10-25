#ifndef MATERIAL_MODEL_LINEAR_ELASTIC
#define MATERIAL_MODEL_LINEAR_ELASTIC

#include "Utilities.h"


//namespace MaterialModels {


class MaterialModelLinearElastic {

public:
  MaterialModelLinearElastic() {}
  MaterialModelLinearElastic(const double & lambda, const double & nu, const double & density, 
    const double & toughness, const double & l0, const size_t & flag) {
    _tangentMatrix.fill(0.);
    _toughness = toughness;
    _l0 = l0;
    _strainHistoryField = 0.;
    double lambda2D = lambda;
    if (flag==0) { // plane strain
      _lambda = lambda;
    }
    else if (flag==1) { // plane stress
      _lambda = 2*lambda*nu/(lambda+2*nu);
    }
    _nu = nu;

    _tangentMatrix << _lambda+2*_nu, _lambda, 0,
                      _lambda, _lambda+2*_nu, 0,
                      0, 0, _nu;
    _density = density;
    _indexContainer.resize(16);
    _indexContainer[0]  << 0, 0, 0, 0, 0, 0;
    _indexContainer[1]  << 0, 0, 0, 1, 0, 2;
    _indexContainer[2]  << 0, 0, 1, 0, 0, 2;
    _indexContainer[3]  << 0, 0, 1, 1, 0, 1;
    _indexContainer[4]  << 0, 1, 0, 0, 2, 0;
    _indexContainer[5]  << 0, 1, 0, 1, 2, 2;
    _indexContainer[6]  << 0, 1, 1, 0, 2, 2;
    _indexContainer[7]  << 0, 1, 1, 1, 2, 1;
    _indexContainer[8]  << 1, 0, 0, 0, 2, 0;
    _indexContainer[9]  << 1, 0, 0, 1, 2, 2;
    _indexContainer[10] << 1, 0, 1, 0, 2, 2;
    _indexContainer[11] << 1, 0, 1, 1, 2, 1;
    _indexContainer[12] << 1, 1, 0, 0, 1, 0;
    _indexContainer[13] << 1, 1, 0, 1, 1, 2;
    _indexContainer[14] << 1, 1, 1, 0, 1, 2;
    _indexContainer[15] << 1, 1, 1, 1, 1, 1;
  }

  
  Vector3d
  computeStress (const Vector3d & strain) const {
    return _tangentMatrix*strain;
  }

  double
  computeStrainEnergyDensity(const Vector3d & strain) const {
    return 0.5*strain.dot(computeStress(strain));
  }

  double
  computeFractureEnergyDensity(const double & phi, const Vector2d & phiGradient) {
    double energyDensity = 0.;
    energyDensity =  phi*phi/2/_l0;
    energyDensity += phiGradient.dot(phiGradient)*_l0/2.;
    energyDensity *= _toughness;
    return energyDensity;
  }

  double
  computeAndUpdateStrainHistoryField (const vector<Vector2d> & princpleStrainsAndVecs, const size_t & updateFlag) {
    Matrix<double, 2, 2> positiveStrain; positiveStrain.fill(0.);
    double positiveStrainEnergy = 0.;
      for (size_t index = 0; index < princpleStrainsAndVecs.size()-1; index++) {
        positiveStrain += (princpleStrainsAndVecs[0](index)+fabs(princpleStrainsAndVecs[0](index)))/2.*
        princpleStrainsAndVecs[index+1]*princpleStrainsAndVecs[index+1].transpose();
      }
    double traceStrain = princpleStrainsAndVecs[0](0)+princpleStrainsAndVecs[0](1);
    Matrix<double, 2, 2> positiveStrainSquare = positiveStrain*positiveStrain;
    positiveStrainEnergy = pow((traceStrain+fabs(traceStrain))/2.,2)*_lambda/2.+_nu*(positiveStrainSquare(0,0)+positiveStrainSquare(1,1));
    if (positiveStrainEnergy > _strainHistoryField) {// && updateFlag == 1) {
      _strainHistoryField = positiveStrainEnergy;
    }
    //return max(positiveStrainEnergy, _strainHistoryField);
    return _strainHistoryField;
  }

  double
  onlyComputeStrainHistoryField (const vector<Vector2d> & princpleStrainsAndVecs) const {
    Matrix<double, 2, 2> positiveStrain; positiveStrain.fill(0.);
    double positiveStrainEnergy = 0.;
    for (size_t index = 0; index < princpleStrainsAndVecs.size()-1; index++) {
        positiveStrain += (princpleStrainsAndVecs[0](index)+fabs(princpleStrainsAndVecs[0](index)))/2.*
        princpleStrainsAndVecs[index+1]*princpleStrainsAndVecs[index+1].transpose();
    }
    double traceStrain = princpleStrainsAndVecs[0](0)+princpleStrainsAndVecs[0](1);
    Matrix<double, 2, 2> positiveStrainSquare = positiveStrain*positiveStrain;
    positiveStrainEnergy = pow((traceStrain+fabs(traceStrain))/2.,2)*_lambda/2.+_nu*(positiveStrainSquare(0,0)+positiveStrainSquare(1,1));
    return max(positiveStrainEnergy, _strainHistoryField);   
  }

  
  Matrix<double, 3, 3>
  computeAndUpdateTangentMatrix(const double & phi, const double & k, 
    const vector<Vector2d> & princpleStrainsAndVecs, const size_t & updateFlag) {
    if (!(princpleStrainsAndVecs[0](0)==0&&princpleStrainsAndVecs[0](1)==0)) {
      double traceStrain = princpleStrainsAndVecs[0](0)+princpleStrainsAndVecs[0](1);
      Matrix<double, 3, 3> dBarMatrix = _lambda*(((1-k)*(1-phi)*(1-phi)+k)*Utilities::Heaviside(traceStrain)
        +Utilities::Heaviside(-1.*traceStrain))*Utilities::getBarDMatrixWithoutCoefficient();
      vector<Matrix<double, 3, 3> > pMatrices = Utilities::getPlusMinusPMatrix(princpleStrainsAndVecs, _indexContainer);
      Matrix<double, 3, 3> dTildaMatrix = 2*_nu*(((1-k)*(1-phi)*(1-phi)+k)*pMatrices[0]+pMatrices[1]);
      dTildaMatrix += dBarMatrix;
      //if (updateFlag == 1) {
        _tangentMatrix = dTildaMatrix;
      //}
      return dTildaMatrix;
    }
    else {
      return _tangentMatrix;
    }
  }
  

  Matrix<double, 3, 3>
  getTangentMatrix() const {
    return _tangentMatrix;
  }

  vector<Matrix<size_t, 6, 1> >
  getIndexContainer() const {
    return _indexContainer;
  }

  void
  changeDensity(const double & density) {
    _density = density;
  }

  double
  getDensity() const {
    return _density;
  }

  double
  getToughness() const {
    return _toughness;
  }

  double
  getL0() const {
    return _l0;
  }

  double
  getLambda() const {
    return _lambda;
  }

  double
  getNu() const {
    return _nu;
  }

  double
  getStrainHistoryField() const {
    return _strainHistoryField;
  }

private:
  Matrix<double, 3, 3>            _tangentMatrix;
  double                          _density;
  double                          _toughness;
  double                          _l0;
  double                          _lambda;
  double                          _nu;
  double                          _strainHistoryField;
  vector<Matrix<size_t, 6, 1> >   _indexContainer;
};

//} // MaterialModels
#endif // MATERIAL_MODEL_LINEAR_ELASTIC