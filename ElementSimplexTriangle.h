// -*- C++ -*-
#ifndef ELEMENT_TRIANGLE
#define ELEMENT_TRIANGLE


#include "MaterialModelLinearElastic.h"


//namespace Elements2D {


class SimplexTriangle {
public:
  SimplexTriangle () {}
  SimplexTriangle(const vector<Vector2d> & nodePositions, const vector<size_t> & nodeIds,
                  vector<MaterialModelLinearElastic*> materialmodel, const size_t & qpnumber) :
                  _nodeIds(nodeIds), _qpnumber(qpnumber), _materialmodel(materialmodel) {

    _quadpoints.resize(qpnumber);
    _quadweights.resize(qpnumber);
    if (qpnumber == 1) {
      _quadweights[0] = 0.5;
      _quadpoints[0] = Vector2d(1./3.,1./3.);
    }
    if (qpnumber == 3) {
      _quadweights[0] = 1./6.;
      _quadweights[1] = 1./6.;
      _quadweights[2] = 1./6.;

      _quadpoints[0] = Vector2d(1./6.,1./6.);
      _quadpoints[1] = Vector2d(2./3.,1./6.);
      _quadpoints[2] = Vector2d(1./6.,2./3.);
    }

    Matrix<double, 3, 4> derivativeToStrains;
    derivativeToStrains.fill(0.);
    derivativeToStrains(0,0) = 1;
    derivativeToStrains(1,3) = 1;
    derivativeToStrains(2,1) = 1;
    derivativeToStrains(2,2) = 1;

    _bMatrices.resize(qpnumber);
    _bPhiMatrices.resize(qpnumber);
    _weightedJacobians.resize(qpnumber);
    _nodalWeights.resize(nodeIds.size());
    _shapeFunctionPhi.resize(qpnumber);
    _shapeFunctionDisplacement.resize(qpnumber);

    Matrix<double, 4, 6> shapeFunctionDerivatives; shapeFunctionDerivatives.fill(0.);
    Matrix<double, 2, 2> jacobian; jacobian.fill(0.);
    Matrix<double, 4, 4> blockGamma; blockGamma.fill(0.);
    Matrix<double, 2, 3> shapeFunctionDerivativesPhi; shapeFunctionDerivativesPhi.fill(0.);
    double volume = 0.;

    for (size_t qpindex = 0; qpindex < qpnumber; qpindex++) {
      // local derivative of shape function for CST is constant, independent of qp value
      shapeFunctionDerivatives.fill(0.);
      shapeFunctionDerivatives(0,0) = 1; shapeFunctionDerivatives(0,4) = -1;
      shapeFunctionDerivatives(1,2) = 1; shapeFunctionDerivatives(1,4) = -1;
      shapeFunctionDerivatives(2,1) = 1; shapeFunctionDerivatives(2,5) = -1;
      shapeFunctionDerivatives(3,3) = 1; shapeFunctionDerivatives(3,5) = -1;

      shapeFunctionDerivativesPhi(0,0) = 1; shapeFunctionDerivativesPhi(0,2) = -1;
      shapeFunctionDerivativesPhi(1,1) = 1; shapeFunctionDerivativesPhi(1,2) = -1; 

      // jacobian is also constant for CST, indepedent of qp value

      jacobian(0,0) = nodePositions[0](0) - nodePositions[2](0);
      jacobian(0,1) = nodePositions[0](1) - nodePositions[2](1);
      jacobian(1,0) = nodePositions[1](0) - nodePositions[2](0);
      jacobian(1,1) = nodePositions[1](1) - nodePositions[2](1);


      // blockGamma, accordingly, is also constant
      blockGamma.fill(0.);
      blockGamma.block<2,2>(0,0) = jacobian.inverse();
      blockGamma.block<2,2>(2,2) = jacobian.inverse();

      _bMatrices[qpindex] = derivativeToStrains*blockGamma*shapeFunctionDerivatives;
      _bPhiMatrices[qpindex] = jacobian.inverse()*shapeFunctionDerivativesPhi;
      _weightedJacobians[qpindex] = _quadweights[qpindex]*jacobian.determinant()*1.; //1. is the thickness
      volume += _weightedJacobians[qpindex];

      _shapeFunctionPhi[qpindex](0,0) = _quadpoints[qpindex](0);
      _shapeFunctionPhi[qpindex](0,1) = _quadpoints[qpindex](1);
      _shapeFunctionPhi[qpindex](0,2) = 1.-_quadpoints[qpindex](0)-_quadpoints[qpindex](1);

      _shapeFunctionDisplacement[qpindex].fill(0.);
      _shapeFunctionDisplacement[qpindex](0,0) = _quadpoints[qpindex](0);
      _shapeFunctionDisplacement[qpindex](0,2) = _quadpoints[qpindex](1);
      _shapeFunctionDisplacement[qpindex](0,4) = 1.-_quadpoints[qpindex](0)-_quadpoints[qpindex](1);
      _shapeFunctionDisplacement[qpindex](1,1) = _quadpoints[qpindex](0);
      _shapeFunctionDisplacement[qpindex](1,3) = _quadpoints[qpindex](1);
      _shapeFunctionDisplacement[qpindex](1,5) = 1.-_quadpoints[qpindex](0)-_quadpoints[qpindex](1);
    }

    for (size_t index = 0; index < _nodeIds.size(); index++) {
      _nodalWeights[index] = volume/double(_nodeIds.size());
    }
    _volume = volume;

  }

  double
  computeStrainEnergy(const vector<Vector3d> & strainsAtGaussianPointsTotal) const {
    double energy = 0.0;
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      const Vector3d strain = strainsAtGaussianPointsTotal[qpindex];
      energy += _weightedJacobians[qpindex] * _materialmodel[qpindex]->computeStrainEnergyDensity(strain);
    }
    return energy;
  }

  double
  computeFractureEnergy(const vector<double> & nodalPhisTotal, const vector<double> & phisAtGaussianPointsTotal) const {

    vector<Vector2d> phiGradientsAtGaussianPointsTotal = computePhiGradientAtGaussianPoints(nodalPhisTotal);
    double fractureEnergy = 0.;
    double phi = 0;
    Vector2d phiGradient;
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      phi = phisAtGaussianPointsTotal[qpindex];
      phiGradient = phiGradientsAtGaussianPointsTotal[qpindex];
      fractureEnergy += _weightedJacobians[qpindex] * _materialmodel[qpindex]->computeFractureEnergyDensity(phi, phiGradient);
    }
    return fractureEnergy;
  }

  vector<Vector2d>
  computeForces(const vector<Vector2d> & nodalDisplacementsTotal) const {
    Matrix<double, 6, 1> nodalForcesVector;
    nodalForcesVector.fill(0.);
    Vector3d stress; stress.fill(0.);
    vector<Vector3d> strainsAtGaussianPoints = computeStrainsAtGaussianPoints(nodalDisplacementsTotal);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      stress = _materialmodel[qpindex]->computeStress(strainsAtGaussianPoints[qpindex]);
      nodalForcesVector += _weightedJacobians[qpindex]*_bMatrices[qpindex].transpose()*stress;
    }
    vector<Vector2d> nodalForces(3);
    nodalForces[0] = nodalForcesVector.block<2,1>(0,0);
    nodalForces[1] = nodalForcesVector.block<2,1>(2,0);
    nodalForces[2] = nodalForcesVector.block<2,1>(4,0);
    return nodalForces;
  }

  vector<Vector2d>
  computeBodyForces(const Vector2d & acceleration) const {
    vector<Vector2d> nodeForces(3);
    nodeForces[0].fill(0.);
    nodeForces[1].fill(0.);
    nodeForces[2].fill(0.);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      nodeForces[0](0) += _materialmodel[qpindex]->getDensity()*acceleration(0)*_weightedJacobians[qpindex]*_quadpoints[qpindex](0);
      nodeForces[0](1) += _materialmodel[qpindex]->getDensity()*acceleration(1)*_weightedJacobians[qpindex]*_quadpoints[qpindex](0);
      nodeForces[1](0) += _materialmodel[qpindex]->getDensity()*acceleration(0)*_weightedJacobians[qpindex]*_quadpoints[qpindex](1);
      nodeForces[1](1) += _materialmodel[qpindex]->getDensity()*acceleration(1)*_weightedJacobians[qpindex]*_quadpoints[qpindex](1);
      nodeForces[2](0) += _materialmodel[qpindex]->getDensity()*acceleration(0)*_weightedJacobians[qpindex]*(1-_quadpoints[qpindex](0)-_quadpoints[qpindex](1));
      nodeForces[2](1) += _materialmodel[qpindex]->getDensity()*acceleration(1)*_weightedJacobians[qpindex]*(1-_quadpoints[qpindex](0)-_quadpoints[qpindex](1));
    }
    return nodeForces;
  }

  vector<double>
  computePhiBodyForce(const vector<Vector2d> & nodalDisplacementsIncrements, 
    const vector<Vector3d> & strainsAtGaussianPointsAccumulates, const size_t & k) const {
    vector<double> nodePhiForce(3);
    nodePhiForce[0] = 0.;
    nodePhiForce[1] = 0.;
    nodePhiForce[2] = 0.;
    vector<Vector3d> strainsAtGaussianPointsTotal = computeStrainsAtGaussianPoints(nodalDisplacementsIncrements);
    double strainHistoryField = 0.;
    vector<Vector2d> principleStrainsAndVecs(3);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      strainsAtGaussianPointsTotal[qpindex] += strainsAtGaussianPointsAccumulates[qpindex];
      principleStrainsAndVecs = Utilities::computePrincipleStrainsAndVecs(strainsAtGaussianPointsTotal[qpindex]);
      strainHistoryField = _materialmodel[qpindex]->onlyComputeStrainHistoryField(principleStrainsAndVecs);
      nodePhiForce[0] += 2*(1-k)*strainHistoryField*_weightedJacobians[qpindex]*_shapeFunctionPhi[qpindex](0,0);
      nodePhiForce[1] += 2*(1-k)*strainHistoryField*_weightedJacobians[qpindex]*_shapeFunctionPhi[qpindex](0,1);
      nodePhiForce[2] += 2*(1-k)*strainHistoryField*_weightedJacobians[qpindex]*_shapeFunctionPhi[qpindex](0,2);
    }
    return nodePhiForce;
  }

  Matrix<double, 6, 6>
  computeStiffnessMatrix(const vector<double> & nodalPhisIncrements, const vector<Vector2d> & nodalDisplacementsIncrements,
    const vector<double> & phisAtGaussianPointsAccumulates, const vector<Vector3d> & strainsAtGaussianPointsAccumulates, 
    const double & k, const size_t & updateFlag)  const {  
    vector<Vector3d> strainsAtGaussianPointsTotal = computeStrainsAtGaussianPoints(nodalDisplacementsIncrements);
    vector<double> phisAtGaussianPointsTotal = computePhisAtGaussianPoints(nodalPhisIncrements);
    Matrix<double, 6, 6> stiffnessMatrix; stiffnessMatrix.fill(0);
    Matrix<double, 3, 3> TangentMatrix; TangentMatrix.fill(0.);
    vector<Vector2d> principleStrainsAndVecs(3);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      strainsAtGaussianPointsTotal[qpindex] += strainsAtGaussianPointsAccumulates[qpindex];
      phisAtGaussianPointsTotal[qpindex] += phisAtGaussianPointsAccumulates[qpindex];
      principleStrainsAndVecs = Utilities::computePrincipleStrainsAndVecs(strainsAtGaussianPointsTotal[qpindex]);
      TangentMatrix = _materialmodel[qpindex]->computeAndUpdateTangentMatrix(phisAtGaussianPointsTotal[qpindex], k, principleStrainsAndVecs, updateFlag);
      stiffnessMatrix += _weightedJacobians[qpindex]*_bMatrices[qpindex].transpose()*TangentMatrix*_bMatrices[qpindex];
    }
    return stiffnessMatrix;
  }

  Matrix<double, 6, 6>
  computeSimpleStiffnessMatrix() {
    Matrix<double, 6, 6> stiffnessMatrix;
    stiffnessMatrix.fill(0.);
    Matrix<double, 3, 3> TangentMatrix;
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      TangentMatrix = _materialmodel[qpindex]->getTangentMatrix();
      stiffnessMatrix += _weightedJacobians[qpindex]*_bMatrices[qpindex].transpose()*TangentMatrix*_bMatrices[qpindex];
    }
    return stiffnessMatrix;
  }

  Matrix<double, 6, 6>
  computeConsistentMassMatrix() const {
    Matrix<double, 6, 6> consistentMassMatrix;
    consistentMassMatrix.fill(0.);

    /*
    double volume = 0;
    for (unsigned int qpindex = 0; qpindex < _qpnumber; qpindex++) {
      volume += _weightedJacobians[qpindex];
    }
    volume *= _materialmodel[0]->getDensity()/12.;
    consistentMassMatrix << 2, 0, 1, 0, 1, 0,
                            0, 2, 0, 1, 0, 1,
                            1, 0, 2, 0, 1, 0,
                            0, 1, 0, 2, 0, 1,
                            1, 0, 1, 0, 2, 0,
                            0, 1, 0, 1, 0, 2;
    consistentMassMatrix *= volume;
    */
    
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      consistentMassMatrix += _weightedJacobians[qpindex]*_shapeFunctionDisplacement[qpindex].transpose()*_shapeFunctionDisplacement[qpindex];
    }
    consistentMassMatrix *= _materialmodel[0]->getDensity();
    
    return consistentMassMatrix;
  }

  Matrix<double, 3, 3>
  computePhiStiffnessMatrix(const vector<Vector2d> & nodalDisplacementsIncrements,
    const vector<Vector3d> & strainsAtGaussianPointsAccumulates,
    const double & k, const size_t & updateFlag) {
    vector<Vector3d> strainsAtGaussianPointsTotal = computeStrainsAtGaussianPoints(nodalDisplacementsIncrements);
    Matrix<double, 3, 3> bMatrix; bMatrix.fill(0.);
    Matrix<double, 3, 3> nMatrix; nMatrix.fill(0.);
    double strainHistoryField = 0.;
    vector<Vector2d> principleStrainsAndVecs(3);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      strainsAtGaussianPointsTotal[qpindex] += strainsAtGaussianPointsAccumulates[qpindex];
      principleStrainsAndVecs = Utilities::computePrincipleStrainsAndVecs(strainsAtGaussianPointsTotal[qpindex]);
      strainHistoryField = _materialmodel[qpindex]->computeAndUpdateStrainHistoryField(principleStrainsAndVecs, updateFlag);
      bMatrix += _weightedJacobians[qpindex]*_materialmodel[qpindex]->getToughness()*_materialmodel[qpindex]->getL0()*_bPhiMatrices[qpindex].transpose()*_bPhiMatrices[qpindex];
      nMatrix += (_materialmodel[qpindex]->getToughness()/_materialmodel[qpindex]->getL0()+2*(1-k)*strainHistoryField)*_weightedJacobians[qpindex]*_shapeFunctionPhi[qpindex].transpose()*_shapeFunctionPhi[qpindex];
    }
    return bMatrix+nMatrix;
  }

  Matrix<double, 3, 3>
  computePhiStiffnessMatrixBpart() {
    Matrix<double, 3, 3> bMatrix; bMatrix.fill(0.);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      bMatrix += _weightedJacobians[qpindex]*_materialmodel[qpindex]->getToughness()*_materialmodel[qpindex]->getL0()*_bPhiMatrices[qpindex].transpose()*_bPhiMatrices[qpindex];
    }
    return bMatrix;
  }


  
  Matrix<double, 3, 3>
  computePhiStiffnessMatrixNpart(const vector<Vector2d> & nodalDisplacementsIncrements,
    const vector<Vector3d> & strainsAtGaussianPointsAccumulates,
    const double & k) {
    vector<Vector3d> strainsAtGaussianPointsTotal = strainsAtGaussianPointsAccumulates;
    Matrix<double, 3, 3> nMatrix; nMatrix.fill(0.);
    double strainHistoryField = 0.;
    vector<Vector2d> principleStrainsAndVecs(3);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      strainsAtGaussianPointsTotal[qpindex] += computeStrainsAtGaussianPoints(nodalDisplacementsIncrements)[qpindex];
      principleStrainsAndVecs = Utilities::computePrincipleStrainsAndVecs(strainsAtGaussianPointsTotal[qpindex]);
      strainHistoryField = _materialmodel[qpindex]->onlyComputeStrainHistoryField(principleStrainsAndVecs);
      //cout << "the strain history field is "  << strainHistoryField << endl;
      //cout << "Gc/l0 is "  << _materialmodel[qpindex]->getToughness()/l0 << endl;
      nMatrix += (_materialmodel[qpindex]->getToughness()/_materialmodel[qpindex]->getL0()+2*(1-k)*strainHistoryField)*_weightedJacobians[qpindex]*_shapeFunctionPhi[qpindex].transpose()*_shapeFunctionPhi[qpindex];
      //nMatrix += _weightedJacobians[qpindex]*_shapeFunctionPhi[qpindex].transpose()*_shapeFunctionPhi[qpindex];
    }
    return nMatrix;
  }


  vector<Vector3d>
  computeStrainsAtGaussianPoints (const vector<Vector2d> & nodalDisplacements) const {
    vector<Vector3d> strainsAtGaussianPoints(_qpnumber);
    Matrix<double, 6, 1> displacementVector; displacementVector.fill(0.); 
    displacementVector.block<2,1>(0,0) = nodalDisplacements[0];
    displacementVector.block<2,1>(2,0) = nodalDisplacements[1];
    displacementVector.block<2,1>(4,0) = nodalDisplacements[2];

    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      strainsAtGaussianPoints[qpindex] = _bMatrices[qpindex]*displacementVector;
    }
    return strainsAtGaussianPoints;
  }

  vector<Vector3d>
  computeStressesAtGaussianPoints(const vector<Vector2d> & nodalDisplacements) const {
    vector<Vector3d> stresses(_qpnumber);
    vector<Vector3d> strainsAtGaussianPoints = computeStrainsAtGaussianPoints(nodalDisplacements);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      stresses[qpindex] = _materialmodel[qpindex]->computeStress(strainsAtGaussianPoints[qpindex]);
    }
    return stresses;
  }

  vector<double>
  computePhisAtGaussianPoints(const vector<double> & Phis) const {
    vector<double> gaussianPhis(_qpnumber);
    Vector3d nodalPhis;
    for (size_t pindex = 0; pindex < nodalPhis.size(); pindex++) {
      nodalPhis(pindex) = Phis[pindex];
    }
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      gaussianPhis[qpindex] = _shapeFunctionPhi[qpindex]*nodalPhis;
    }
    return gaussianPhis;
  }

  vector<Vector2d>
  computePhiGradientAtGaussianPoints(const vector<double> & Phis) const {
    Vector3d nodalPhis;
    vector<Vector2d> gaussianPhiGradient(_qpnumber);
    for (size_t pindex = 0; pindex < nodalPhis.size(); pindex++) {
      nodalPhis(pindex) = Phis[pindex];
    }
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      gaussianPhiGradient[qpindex] = _bPhiMatrices[qpindex]*nodalPhis;
    }
    return gaussianPhiGradient;
  }

  
  vector<double>
  computeEnergyDensityAtGaussianPoints(const vector<Vector2d> & nodalDisplacements) const {
    vector<double> gaussianEnergyDensities(_qpnumber);
    vector<Vector3d> strainsAtGaussianPoints = computeStrainsAtGaussianPoints(nodalDisplacements);
    for (size_t qpindex = 0; qpindex < _qpnumber; qpindex++) {
      const Vector3d strain = strainsAtGaussianPoints[qpindex];
      gaussianEnergyDensities[qpindex] = _materialmodel[qpindex]->computeStrainEnergyDensity(strain);
    }
    return gaussianEnergyDensities;
  }
  

  vector<size_t>
  getNodeIds() const {
    return _nodeIds;
  }

  vector<double>
  getNodalWeights() const {
    return _nodalWeights;
  }

  vector<double>
  getElementStrainHistory() const {
    vector<double> strainHistoryField(_qpnumber);
    for (unsigned int qpindex = 0; qpindex < strainHistoryField.size(); qpindex++) {
      strainHistoryField[qpindex] = _materialmodel[qpindex]->getStrainHistoryField();
    }
    return strainHistoryField;
  }

  double
  getQPNumber() const {
    return _qpnumber;
  }

  double
  getVolume() const {
    return _volume;
  }

  vector<MaterialModelLinearElastic*>
  getMaterialModels() const {
    return _materialmodel;
  }  


private:

  vector<size_t>                                       _nodeIds;
  size_t                                               _qpnumber;
  vector<MaterialModelLinearElastic*>                  _materialmodel;
  vector<double>                                       _weightedJacobians;
  vector<double>                                       _nodalWeights;
  vector<Matrix<double, 3, 6> >                        _bMatrices;
  vector<Matrix<double, 2, 3> >                        _bPhiMatrices;
  vector<Vector2d>                                     _quadpoints;
  vector<double>                                       _quadweights;
  vector<Matrix<double, 1, 3> >                        _shapeFunctionPhi;
  vector<Matrix<double, 2, 6> >                        _shapeFunctionDisplacement;
  double                                               _volume;

};

//}

#endif //ELEMENT_TRIANGLE

