// -*- C++ -*-
#ifndef WORLDYNAMIC_H
#define WORLDDYNAMIC_H

#include "MaterialModelLinearElastic.h"
#include "ElementSimplexTriangle.h"

class WorldDynamic {
public:
  WorldDynamic () {}
  WorldDynamic (const vector<SimplexTriangle> & elements, const size_t numberOfNodes, 
    const vector<Vector2d> & nodePositions, const vector<Vector2d> & nodeVelocities,
    const vector<Vector2d> & nodeAccelerations,
    const vector<double> & nodePhis, const double & k, const int & size) {
    _elements = elements;
    _numberOfNodes = numberOfNodes;
    _numberOfElements = elements.size();
    _nodePositions = nodePositions;
    _nodeVelocities = nodeVelocities;
    _nodeAccelerations = nodeAccelerations;
    _nodePhis = nodePhis;
    _k = k;

    _elementChunkSize.resize(size);
    _elementOffsetSize.resize(size);

    _displacementMatrixChunkSize.resize(size);
    _displacementMatrixOffsetSize.resize(size);

    _phiMatrixChunkSize.resize(size);
    _phiMatrixOffsetSize.resize(size);

    _displacementStartingRowIndex.resize(size);
    _displacementNumberOfRow.resize(size);

    _phiStartingRowIndex.resize(size);
    _phiNumberOfRow.resize(size);

    _totalDisplacementMatrixNumber = 0;
    _totalPhiMatrixNumber = 0;

    _nodeBasedDisplacementStartingIndex.resize(size);
    _nodeBasedDisplacementChunkSize.resize(size);

    vector<int> nodeChunkSize(size);


    const size_t singleElementChunkSize = _elements.size()/size;
    const size_t singleNodeChunkSize = _numberOfNodes/size;
    const size_t overHead = _elements.size() - singleElementChunkSize*size;
    const size_t nodeOverHead = _numberOfNodes - singleNodeChunkSize*size;

    const size_t displacementSingleElementMatrixNumber = 36;
    const size_t phiSingleElementMatrixNumber = 9;
    const size_t displacementSingleNodeVectorNumber = 2;

    const size_t singleDisplacementRowNumberPerTask = numberOfNodes*2/size;
    const size_t displacementRowNumberOverHead = numberOfNodes*2 - singleDisplacementRowNumberPerTask*size;
    const size_t singlePhiRowNumberPerTask = numberOfNodes/size;
    const size_t phiRowNumberOverHead = numberOfNodes - singlePhiRowNumberPerTask*size;


    for (size_t index = 0; index < size; index++) {
      if (index < size-1) {
        _elementChunkSize[index] = singleElementChunkSize;
        nodeChunkSize[index] = singleNodeChunkSize;
        _displacementNumberOfRow[index] = singleDisplacementRowNumberPerTask;
        _phiNumberOfRow[index] = singlePhiRowNumberPerTask;
      }
      else {
        _elementChunkSize[index] = singleElementChunkSize+overHead;
        nodeChunkSize[index] = singleNodeChunkSize+nodeOverHead;
        _displacementNumberOfRow[index] = singleDisplacementRowNumberPerTask+displacementRowNumberOverHead;
        _phiNumberOfRow[index] = singlePhiRowNumberPerTask+phiRowNumberOverHead;

      }
      _displacementMatrixChunkSize[index] = _elementChunkSize[index]*displacementSingleElementMatrixNumber;
      _phiMatrixChunkSize[index] = _elementChunkSize[index]*phiSingleElementMatrixNumber;
      _nodeBasedDisplacementChunkSize[index] = nodeChunkSize[index]*displacementSingleNodeVectorNumber;
      
      if (index == 0) {
        _elementOffsetSize[index] = 0;
        _displacementMatrixOffsetSize[index] = 0;
        _phiMatrixOffsetSize[index] = 0;
        _displacementStartingRowIndex[index] = 0;
        _phiStartingRowIndex[index] = 0;
        _nodeBasedDisplacementStartingIndex[index] = 0;
      }
      else {
        _elementOffsetSize[index] = _elementOffsetSize[index-1]+_elementChunkSize[index-1];
        _displacementMatrixOffsetSize[index] = _displacementMatrixOffsetSize[index-1]+_displacementMatrixChunkSize[index-1];
        _phiMatrixOffsetSize[index] = _phiMatrixOffsetSize[index-1]+_phiMatrixChunkSize[index-1];
        _displacementStartingRowIndex[index] = _displacementStartingRowIndex[index-1]+_displacementNumberOfRow[index-1];
        _phiStartingRowIndex[index] = _phiStartingRowIndex[index-1]+_phiNumberOfRow[index-1];
        _nodeBasedDisplacementStartingIndex[index] = _nodeBasedDisplacementStartingIndex[index-1]+_nodeBasedDisplacementChunkSize[index-1];
      }

      _totalDisplacementMatrixNumber += _displacementMatrixChunkSize[index];
      _totalPhiMatrixNumber += _phiMatrixChunkSize[index];
    }
  }

  double
  assembleStrainEnergy(const vector<vector<Vector3d> > & worldStrainsAtGaussianPointsTotal) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double energy = 0.;
    double energyPerTask = 0.;

    #pragma omp parallel for reduction(+:energyPerTask)
    for(size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
      energyPerTask += _elements[eindex].computeStrainEnergy(worldStrainsAtGaussianPointsTotal[eindex]);
    }
    MPI_Allreduce(&energyPerTask, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return energy;
  }

  double
  assembleFractureEnergy(const vector<double> & worldPhisTotal, 
    const vector< vector<double> > & worldPhisAtGaussianPointsTotal) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double energy = 0.;
    double energyPerTask = 0.;

    vector<size_t> elementNodeIds;
    vector<double> elementPhisTotal;

    #pragma omp parallel for schedule(static) private(elementNodeIds, elementPhisTotal)
    for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
      elementNodeIds = _elements[eindex].getNodeIds();
      elementPhisTotal = Utilities::getElementPhisFromGlobalList(elementNodeIds, worldPhisTotal);
      energyPerTask += _elements[eindex].computeFractureEnergy(elementPhisTotal, worldPhisAtGaussianPointsTotal[eindex]);
    }
    MPI_Allreduce(&energyPerTask, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return energy;
  }


  VectorXd
  assembleInternalForceVector(const vector<Vector2d> & displacements) {

    const size_t numberDofs = displacements.size()*2;
    VectorXd forceVector(numberDofs);
    forceVector.fill(0.);

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacements(3);
    vector<Vector2d> elementForces;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VectorXd forceVectorPerTask = forceVector;

    #pragma omp parallel private(elementNodeIds, elementDisplacements, elementForces)
    {
      VectorXd threadForceVector(numberDofs); threadForceVector.fill(0.);
      #pragma omp for schedule(static)
      for(unsigned int eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        elementNodeIds = _elements[eindex].getNodeIds();
        elementDisplacements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, displacements);
        elementForces = _elements[eindex].computeForces(elementDisplacements);
        for(unsigned int nindex = 0; nindex < elementForces.size(); nindex++) {
          for(unsigned int i = 0; i < 2; i++) {
             threadForceVector(elementNodeIds[nindex]*2+i) += elementForces[nindex](i);
          }
        }
      }
      #pragma omp critical
      {
        forceVectorPerTask += threadForceVector;
      }
    }
    MPI_Allreduce(forceVectorPerTask.data(), forceVector.data(), numberDofs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return forceVector;
  }

  VectorXd
  assembleBodyForceVectorPerTask(const Vector2d & acceleration) {

    const size_t numberDofs = _numberOfNodes*2;
    //VectorXd forceVector(numberDofs);
    //forceVector.fill(0.);

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementForces;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VectorXd forceVectorPerTask(numberDofs); forceVectorPerTask.fill(0.);

    #pragma omp parallel private(elementNodeIds, elementForces)
    {
      VectorXd threadForceVector(numberDofs); threadForceVector.fill(0.);
      #pragma omp for schedule(static)
      for(unsigned int eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        elementNodeIds = _elements[eindex].getNodeIds();
        elementForces = _elements[eindex].computeBodyForces(acceleration);
        for(unsigned int nindex = 0; nindex < elementForces.size(); nindex++) {
          for(unsigned int i = 0; i < 2; i++) {
             threadForceVector(elementNodeIds[nindex]*2+i) += elementForces[nindex](i);
          }
        }
      }
      #pragma omp critical
      {
        forceVectorPerTask += threadForceVector;
      }
    }
    //MPI_Allreduce(forceVectorPerTask.data(), forceVector.data(), numberDofs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return forceVectorPerTask;
  }

  VectorXd
  assemblePhiBodyForceVectorPerTask (const vector<Vector2d> & worldDisplacementsIncrements,
    const vector<vector<Vector3d> > & worldStrainsAtGaussianPointsAccumulates) {

    const size_t numberDofs = _numberOfNodes;
    //VectorXd forceVector(numberDofs);
    //forceVector.fill(0.);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VectorXd forceVectorPerTask(numberDofs); forceVectorPerTask.fill(0.);

    vector<size_t> elementNodeIds;
    vector<double> elementForces;
    vector<Vector2d> elementDisplacementsIncrements;

    #pragma omp parallel private(elementNodeIds, elementForces, elementDisplacementsIncrements)
    {
      VectorXd threadForceVector(numberDofs); threadForceVector.fill(0.);
      #pragma omp for schedule(static)
      for (unsigned int eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        elementNodeIds = _elements[eindex].getNodeIds();
        elementDisplacementsIncrements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, worldDisplacementsIncrements);
        elementForces = _elements[eindex].computePhiBodyForce(elementDisplacementsIncrements, worldStrainsAtGaussianPointsAccumulates[eindex], _k);
        for (unsigned int nindex = 0; nindex < elementForces.size(); nindex++) {
          threadForceVector(elementNodeIds[nindex]) += elementForces[nindex];
        }
      }
      #pragma omp critical
      {
        forceVectorPerTask += threadForceVector;
      }      
    }
    //MPI_Allreduce(forceVectorPerTask.data(), forceVector.data(), numberDofs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return forceVectorPerTask;
  }

  VectorXd
  assembleStiffnessMatrixAndDiagonalEntriesPerTask(const vector<double> & worldPhisIncrements, const vector<Vector2d> & worldDisplacementsIncrements, 
    const vector< vector<double> > & worldPhisAtGaussianPointsAccumulates, const vector<vector<Vector3d> > & worldStrainsAtGaussianPointsAccumulates,
    const size_t & updateFlag, SparseMatrix<double, RowMajor> & worldDisplacementStiffnessMatrixPerTask) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //int indexChunkSize[_displacementMatrixChunkSize.size()];
    //int indexOffsetSize[_displacementMatrixOffsetSize.size()];
    //copy(_displacementMatrixChunkSize.begin(), _displacementMatrixChunkSize.end(), indexChunkSize);
    //copy(_displacementMatrixOffsetSize.begin(), _displacementMatrixOffsetSize.end(), indexOffsetSize);

    size_t rowIndex = 0;
    size_t colIndex = 0;
    const size_t displacementSingleElementMatrixNumber = 36;

    //vector<Triplet<double, size_t> > tripletList(_totalDisplacementMatrixNumber);
    vector<Triplet<double, size_t> > tripletList(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);

    VectorXd diagonalEntriesPerTask(_numberOfNodes*2); diagonalEntriesPerTask.fill(0.);

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacementsIncrements;
    vector<double> elementPhisIncrements;
    Matrix<double, 6, 6> elementStiffnessMatrix; elementStiffnessMatrix.fill(0.);
    size_t nodeId1;
    size_t nodeId2;


    //vector<size_t> RowIndexPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    //vector<size_t> ColIndexPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    //vector<double> ValueRefPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);

    //vector<size_t> worldRowIndex(_totalDisplacementMatrixNumber);
    //vector<size_t> worldColIndex(_totalDisplacementMatrixNumber);
    //vector<double> worldValueRef(_totalDisplacementMatrixNumber);

    #pragma omp parallel private(rowIndex, colIndex, elementNodeIds, elementDisplacementsIncrements, elementPhisIncrements, elementStiffnessMatrix, nodeId1, nodeId2)
    {
      size_t localElementIndex = 0;
      size_t localInsertIndex = 0;
      //VectorXd diagonalEntriesPerThread = diagonalEntriesPerTask; diagonalEntriesPerThread.fill(0.);
      #pragma omp for schedule(static)
      for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        localElementIndex = eindex - _elementOffsetSize[rank];
        elementNodeIds = _elements[eindex].getNodeIds();
        elementDisplacementsIncrements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, worldDisplacementsIncrements);
        elementPhisIncrements = Utilities::getElementPhisFromGlobalList(elementNodeIds, worldPhisIncrements);
        elementStiffnessMatrix = _elements[eindex].computeStiffnessMatrix(elementPhisIncrements, elementDisplacementsIncrements, 
          worldPhisAtGaussianPointsAccumulates[eindex], worldStrainsAtGaussianPointsAccumulates[eindex], _k, updateFlag);
        for (size_t nodeindex1 = 0; nodeindex1 < elementNodeIds.size(); nodeindex1++) {
          nodeId1 = elementNodeIds[nodeindex1];
          for (size_t nodeindex2 = 0; nodeindex2 < elementNodeIds.size(); nodeindex2++) {
            nodeId2 = elementNodeIds[nodeindex2];
            for (size_t i = 0; i < 2; i++) {
              for (size_t j = 0; j < 2; j++) {
                rowIndex = nodeId1*2+i;
                colIndex = nodeId2*2+j;

                localInsertIndex = elementNodeIds.size()*2*2*nodeindex1+2*2*nodeindex2+2*i+j;

                tripletList[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = 
                  Triplet<double, size_t>(rowIndex, colIndex, elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j));

                //if (colIndex == rowIndex) {
                //  diagonalEntriesPerThread(rowIndex) += elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j);
                //}

                //RowIndexPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = rowIndex;
                //ColIndexPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = colIndex;
                //ValueRefPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j);
              }
            }
          }
        }
      }
      //#pragma omp critical
      //{
      //  diagonalEntriesPerTask += diagonalEntriesPerThread;
      //}
    }
    worldDisplacementStiffnessMatrixPerTask.setFromTriplets(tripletList.begin(), tripletList.end());
    
    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < worldDisplacementStiffnessMatrixPerTask.outerSize(); ++index) {
      for (SparseMatrix<double, RowMajor>::InnerIterator it(worldDisplacementStiffnessMatrixPerTask, index); it; ++it) {
        if (it.col()==index) {
          diagonalEntriesPerTask(index) = it.value();
        }
      }
    }
    
    return diagonalEntriesPerTask;
    /*
    MPI_Allgatherv(RowIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldRowIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ColIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldColIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ValueRefPerTask.data(), indexChunkSize[rank], MPI_DOUBLE, worldValueRef.data(), indexChunkSize, indexOffsetSize, MPI_DOUBLE, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < _totalDisplacementMatrixNumber; index++) {
      tripletList[index] = Triplet<double, size_t>(worldRowIndex[index], worldColIndex[index], worldValueRef[index]);
    }
    return tripletList;
    */
  }

  void
  assembleStiffnessMatrixAsIsPerTask(const vector< vector<double> > & worldPhisAtGaussianPointsAccumulates, 
    const vector<vector<Vector3d> > & worldStrainsAtGaussianPointsAccumulates, SparseMatrix<double, RowMajor> & worldDisplacementStiffnessMatrixPerTask) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //int indexChunkSize[_displacementMatrixChunkSize.size()];
    //int indexOffsetSize[_displacementMatrixOffsetSize.size()];
    //copy(_displacementMatrixChunkSize.begin(), _displacementMatrixChunkSize.end(), indexChunkSize);
    //copy(_displacementMatrixOffsetSize.begin(), _displacementMatrixOffsetSize.end(), indexOffsetSize);

    size_t rowIndex = 0;
    size_t colIndex = 0;
    const size_t updateFlag = 0;
    const size_t displacementSingleElementMatrixNumber = 36;

    //vector<Triplet<double, size_t> > tripletList(_totalDisplacementMatrixNumber);
    vector<Triplet<double, size_t> > tripletList(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacementsIncrements;
    vector<double> elementPhisIncrements;
    Matrix<double, 6, 6> elementStiffnessMatrix; elementStiffnessMatrix.fill(0.);
    size_t nodeId1 = 0;
    size_t nodeId2 = 0;


    //vector<size_t> RowIndexPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    //vector<size_t> ColIndexPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    //vector<double> ValueRefPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);

    //vector<size_t> worldRowIndex(_totalDisplacementMatrixNumber);
    //vector<size_t> worldColIndex(_totalDisplacementMatrixNumber);
    //vector<double> worldValueRef(_totalDisplacementMatrixNumber);

    #pragma omp parallel private(rowIndex, colIndex, elementNodeIds, elementDisplacementsIncrements, elementPhisIncrements, elementStiffnessMatrix, nodeId1, nodeId2)
    {
      size_t localElementIndex = 0;
      size_t localInsertIndex = 0;
      #pragma omp for schedule(static)      
      for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        localElementIndex = eindex - _elementOffsetSize[rank];
        elementNodeIds = _elements[eindex].getNodeIds();
        elementDisplacementsIncrements.resize(elementNodeIds.size());
        elementPhisIncrements.resize(elementNodeIds.size());
        for (unsigned int nindex = 0; nindex < elementNodeIds.size(); nindex++) {
          elementDisplacementsIncrements[nindex].fill(0.);
          elementPhisIncrements[nindex] = 0.;
        }
        elementStiffnessMatrix = _elements[eindex].computeStiffnessMatrix(elementPhisIncrements, elementDisplacementsIncrements, 
          worldPhisAtGaussianPointsAccumulates[eindex], worldStrainsAtGaussianPointsAccumulates[eindex], _k, updateFlag);
        for (size_t nodeindex1 = 0; nodeindex1 < elementNodeIds.size(); nodeindex1++) {
          nodeId1 = elementNodeIds[nodeindex1];
          for (size_t nodeindex2 = 0; nodeindex2 < elementNodeIds.size(); nodeindex2++) {
            nodeId2 = elementNodeIds[nodeindex2];
            for (size_t i = 0; i < 2; i++) {
              for (size_t j = 0; j < 2; j++) {
                rowIndex = nodeId1*2+i;
                colIndex = nodeId2*2+j;

                localInsertIndex = elementNodeIds.size()*2*2*nodeindex1+2*2*nodeindex2+2*i+j;
                tripletList[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] 
                  = Triplet<double, size_t>(rowIndex, colIndex, elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j));

                //RowIndexPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = rowIndex;
                //ColIndexPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = colIndex;
                //ValueRefPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j);
              }
            }
          }
        }
      }
    }
    worldDisplacementStiffnessMatrixPerTask.setFromTriplets(tripletList.begin(), tripletList.end());

    /*
    MPI_Allgatherv(RowIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldRowIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ColIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldColIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ValueRefPerTask.data(), indexChunkSize[rank], MPI_DOUBLE, worldValueRef.data(), indexChunkSize, indexOffsetSize, MPI_DOUBLE, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < _totalDisplacementMatrixNumber; index++) {
      tripletList[index] = Triplet<double, size_t>(worldRowIndex[index], worldColIndex[index], worldValueRef[index]);
    }
    return tripletList;
    */
  }


  vector<Triplet<double,size_t> >
  assembleSimpleStiffnessMatrix() {

    size_t rowIndex = 0;
    size_t colIndex = 0;

    vector<Triplet<double,size_t> > tripletList;
    tripletList.reserve(_numberOfNodes);

    vector<size_t> elementNodeIds;
    Matrix<double, 6, 6> elementStiffnessMatrix; elementStiffnessMatrix.fill(0.);
    size_t nodeId1;
    size_t nodeId2;
    for (size_t eindex = 0; eindex < _elements.size(); eindex++) {
      elementNodeIds = _elements[eindex].getNodeIds();
      elementStiffnessMatrix = _elements[eindex].computeSimpleStiffnessMatrix();
      for (size_t nodeindex1 = 0; nodeindex1 < elementNodeIds.size(); nodeindex1++) {
        nodeId1 = elementNodeIds[nodeindex1];
        for (size_t nodeindex2 = 0; nodeindex2 < elementNodeIds.size(); nodeindex2++) {
          nodeId2 = elementNodeIds[nodeindex2];
          for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
              if (elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j) != 0) {
                rowIndex = nodeId1*2+i;
                colIndex = nodeId2*2+j;
                tripletList.push_back(Triplet<double,size_t>(rowIndex,colIndex,elementStiffnessMatrix(nodeindex1*2+i,nodeindex2*2+j)));
              }
            }
          }
        }
      }
    }
    return tripletList;
  }

  vector<Triplet<double,size_t> >
  assembleSimpleMassMatrix() {

    size_t rowIndex = 0;
    size_t colIndex = 0;

    vector<Triplet<double,size_t> > tripletList;
    tripletList.reserve(_numberOfNodes);

    vector<size_t> elementNodeIds;
    Matrix<double, 6, 6> elementMassMatrix; elementMassMatrix.fill(0.);
    size_t nodeId1;
    size_t nodeId2;
    for (size_t eindex = 0; eindex < _elements.size(); eindex++) {
      elementNodeIds = _elements[eindex].getNodeIds();
      elementMassMatrix = _elements[eindex].computeConsistentMassMatrix();
      for (size_t nodeindex1 = 0; nodeindex1 < elementNodeIds.size(); nodeindex1++) {
        nodeId1 = elementNodeIds[nodeindex1];
        for (size_t nodeindex2 = 0; nodeindex2 < elementNodeIds.size(); nodeindex2++) {
          nodeId2 = elementNodeIds[nodeindex2];
          for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
              if (elementMassMatrix(nodeindex1*2+i,nodeindex2*2+j) != 0) {
                rowIndex = nodeId1*2+i;
                colIndex = nodeId2*2+j;
                tripletList.push_back(Triplet<double,size_t>(rowIndex,colIndex,elementMassMatrix(nodeindex1*2+i,nodeindex2*2+j)));
              }
            }
          }
        }
      }
    }
    return tripletList;
  }


  VectorXd
  assembleMassMatrixAndDiagonalEntriesPerTask(SparseMatrix<double, RowMajor> & worldMassMatrixPerTask) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //int indexChunkSize[_displacementMatrixChunkSize.size()];
    //int indexOffsetSize[_displacementMatrixOffsetSize.size()];
    //copy(_displacementMatrixChunkSize.begin(), _displacementMatrixChunkSize.end(), indexChunkSize);
    //copy(_displacementMatrixOffsetSize.begin(), _displacementMatrixOffsetSize.end(), indexOffsetSize);
    const size_t displacementSingleElementMatrixNumber = 36;

    //vector<Triplet<double, size_t> > tripletList(_totalDisplacementMatrixNumber);
    vector<Triplet<double, size_t> > tripletList(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    VectorXd diagonalEntriesPerTask(_numberOfNodes*2); diagonalEntriesPerTask.fill(0.);

    size_t rowIndex = 0;
    size_t colIndex = 0;   

    vector<size_t> elementNodeIds;
    Matrix<double, 6, 6> elementMassMatrix; elementMassMatrix.fill(0.);
    size_t nodeId1;
    size_t nodeId2;


    //vector<size_t> RowIndexPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    //vector<size_t> ColIndexPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);
    //vector<double> ValueRefPerTask(_elementChunkSize[rank]*displacementSingleElementMatrixNumber);

    //vector<size_t> worldRowIndex(_totalDisplacementMatrixNumber);
    //vector<size_t> worldColIndex(_totalDisplacementMatrixNumber);
    //vector<double> worldValueRef(_totalDisplacementMatrixNumber);    

    #pragma omp parallel private(rowIndex, colIndex, elementNodeIds, elementMassMatrix, nodeId1, nodeId2)
    { 
      size_t localElementIndex = 0;
      size_t localInsertIndex = 0;
      //VectorXd diagonalEntriesPerThread = diagonalEntriesPerTask; diagonalEntriesPerThread.fill(0.);
      #pragma omp for schedule(static)
      for(size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        localElementIndex = eindex - _elementOffsetSize[rank];
        elementMassMatrix = _elements[eindex].computeConsistentMassMatrix();
        elementNodeIds = _elements[eindex].getNodeIds();
        for (size_t nodeindex1 = 0; nodeindex1 < elementNodeIds.size(); nodeindex1++) {
          nodeId1 = elementNodeIds[nodeindex1];
          for (size_t nodeindex2 = 0; nodeindex2 < elementNodeIds.size(); nodeindex2++) {
            nodeId2 = elementNodeIds[nodeindex2];
            for (size_t i = 0; i < 2; i++) {
              for (size_t j = 0; j < 2; j++) {
                rowIndex = nodeId1*2+i;
                colIndex = nodeId2*2+j;
                localInsertIndex = elementNodeIds.size()*2*2*nodeindex1+2*2*nodeindex2+2*i+j;

                tripletList[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] 
                  = Triplet<double, size_t>(rowIndex, colIndex, elementMassMatrix(nodeindex1*2+i,nodeindex2*2+j));

                //if (colIndex == rowIndex) {
                //  diagonalEntriesPerThread(rowIndex) += elementMassMatrix(nodeindex1*2+i,nodeindex2*2+j);
                //}

                //RowIndexPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = rowIndex;
                //ColIndexPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = colIndex;
                //ValueRefPerTask[localElementIndex*displacementSingleElementMatrixNumber+localInsertIndex] = elementMassMatrix(nodeindex1*2+i,nodeindex2*2+j);
              }
            }
          }
        }
      }
      //#pragma omp critical
      //{
      //  diagonalEntriesPerTask += diagonalEntriesPerThread;
      //}
    }
    worldMassMatrixPerTask.setFromTriplets(tripletList.begin(), tripletList.end());
    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < worldMassMatrixPerTask.outerSize(); ++index) {
      for (SparseMatrix<double, RowMajor>::InnerIterator it(worldMassMatrixPerTask, index); it; ++it) {
        if (it.col()==index) {
          diagonalEntriesPerTask(index) = it.value();
        }
      }
    }
    return diagonalEntriesPerTask;
    /*
    MPI_Allgatherv(RowIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldRowIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ColIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldColIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ValueRefPerTask.data(), indexChunkSize[rank], MPI_DOUBLE, worldValueRef.data(), indexChunkSize, indexOffsetSize, MPI_DOUBLE, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < _totalDisplacementMatrixNumber; index++) {
      tripletList[index] = Triplet<double, size_t>(worldRowIndex[index], worldColIndex[index], worldValueRef[index]);
    }
    return tripletList;
    */
  }


  VectorXd
  assemblePhiStiffnessMatrixAndDiagonalEntriesPerTask(const vector<Vector2d> & worldDisplacementsIncrements,
    const vector<vector<Vector3d> > & worldStrainsAtGaussianPointsAccumulates, const size_t & updateFlag,
    SparseMatrix<double, RowMajor> & worldPhiStiffnessMatrixPerTask) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //int indexChunkSize[_phiMatrixChunkSize.size()];
    //int indexOffsetSize[_phiMatrixOffsetSize.size()];
    //copy(_phiMatrixChunkSize.begin(), _phiMatrixChunkSize.end(), indexChunkSize);
    //copy(_phiMatrixOffsetSize.begin(), _phiMatrixOffsetSize.end(), indexOffsetSize);

    size_t rowIndex = 0;
    size_t colIndex = 0;
    const size_t phiSingleElementMatrixNumber = 9;

    //vector<Triplet<double,size_t> > tripletList(_totalPhiMatrixNumber);
    vector<Triplet<double,size_t> > tripletList(_elementChunkSize[rank]*phiSingleElementMatrixNumber);
    VectorXd diagonalEntriesPerTask(_numberOfNodes); diagonalEntriesPerTask.fill(0.);

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacementsIncrements;
    Matrix<double, 3, 3> elementPhiStiffnessMatrix; elementPhiStiffnessMatrix.fill(0.);
    size_t nodeId1;
    size_t nodeId2;


    //vector<size_t> RowIndexPerTask(_elementChunkSize[rank]*phiSingleElementMatrixNumber);
    //vector<size_t> ColIndexPerTask(_elementChunkSize[rank]*phiSingleElementMatrixNumber);
    //vector<double> ValueRefPerTask(_elementChunkSize[rank]*phiSingleElementMatrixNumber);

    //vector<size_t> worldRowIndex(_totalPhiMatrixNumber);
    //vector<size_t> worldColIndex(_totalPhiMatrixNumber);
    //vector<double> worldValueRef(_totalPhiMatrixNumber);

    #pragma omp parallel private(rowIndex, colIndex, elementNodeIds, elementDisplacementsIncrements, elementPhiStiffnessMatrix, nodeId1, nodeId2)
    {
      size_t         localElementIndex = 0;
      size_t         localInsertIndex = 0;
      //VectorXd       diagonalEntriesPerThread = diagonalEntriesPerTask; diagonalEntriesPerThread.fill(0.);
      #pragma omp for schedule(dynamic)
      for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        localElementIndex = eindex - _elementOffsetSize[rank];
        elementNodeIds = _elements[eindex].getNodeIds();
        elementDisplacementsIncrements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, worldDisplacementsIncrements);
        elementPhiStiffnessMatrix = _elements[eindex].computePhiStiffnessMatrix(elementDisplacementsIncrements, 
          worldStrainsAtGaussianPointsAccumulates[eindex], _k, updateFlag);
        for (size_t nodeindex1 = 0; nodeindex1 < elementNodeIds.size(); nodeindex1++) {
          nodeId1 = elementNodeIds[nodeindex1];
          for (size_t nodeindex2 = 0; nodeindex2 < elementNodeIds.size(); nodeindex2++) {
            nodeId2 = elementNodeIds[nodeindex2];
            for (size_t i = 0; i < 1; i++) {
              for (size_t j = 0; j < 1; j++) {
                rowIndex = nodeId1*1+i;
                colIndex = nodeId2*1+j;
                localInsertIndex = elementNodeIds.size()*1*1*nodeindex1+1*1*nodeindex2+1*i+j;

                tripletList[localElementIndex*phiSingleElementMatrixNumber+localInsertIndex] 
                  = Triplet<double, size_t>(rowIndex, colIndex, elementPhiStiffnessMatrix(nodeindex1*1+i,nodeindex2*1+j));

                //if (colIndex == rowIndex) {
                //  diagonalEntriesPerThread(rowIndex) += elementPhiStiffnessMatrix(nodeindex1*1+i,nodeindex2*1+j);
                //}

                //RowIndexPerTask[localElementIndex*phiSingleElementMatrixNumber+localInsertIndex] = rowIndex;
                //ColIndexPerTask[localElementIndex*phiSingleElementMatrixNumber+localInsertIndex] = colIndex;
                //ValueRefPerTask[localElementIndex*phiSingleElementMatrixNumber+localInsertIndex] = elementPhiStiffnessMatrix(nodeindex1*1+i,nodeindex2*1+j);

              }
            }
          }
        }
      }
      //#pragma omp critical
      //{
      //  diagonalEntriesPerTask += diagonalEntriesPerThread;
      //}
    }
    worldPhiStiffnessMatrixPerTask.setFromTriplets(tripletList.begin(), tripletList.end());
    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < worldPhiStiffnessMatrixPerTask.outerSize(); ++index) {
      for (SparseMatrix<double, RowMajor>::InnerIterator it(worldPhiStiffnessMatrixPerTask, index); it; ++it) {
        if (it.col()==index) {
          diagonalEntriesPerTask(index) = it.value();
        }
      }
    }
    return diagonalEntriesPerTask;
    /*
    MPI_Allgatherv(RowIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldRowIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ColIndexPerTask.data(), indexChunkSize[rank], my_MPI_SIZE_T, worldColIndex.data(), indexChunkSize, indexOffsetSize, my_MPI_SIZE_T, MPI_COMM_WORLD);
    MPI_Allgatherv(ValueRefPerTask.data(), indexChunkSize[rank], MPI_DOUBLE, worldValueRef.data(), indexChunkSize, indexOffsetSize, MPI_DOUBLE, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < _totalPhiMatrixNumber; index++) {
      tripletList[index] = Triplet<double, size_t>(worldRowIndex[index], worldColIndex[index], worldValueRef[index]);
    }   
    return tripletList;
    */
  }

  vector<Vector3d>
  computeElementStresses(const vector<Vector2d> & displacements) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<Vector3d> allElementAvgStresses; allElementAvgStresses.resize(_elements.size());

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacements;
    vector<Vector3d> elementStresses;
    Vector3d avgStress; avgStress.fill(0.);

    #pragma omp parallel for schedule(static) private(elementNodeIds,elementDisplacements,elementStresses,avgStress)    
    for (size_t elementIndex = _elementOffsetSize[rank]; elementIndex < _elementOffsetSize[rank]+_elementChunkSize[rank]; elementIndex++) { 
      elementNodeIds = _elements[elementIndex].getNodeIds();
      elementDisplacements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, displacements);
      elementStresses = _elements[elementIndex].computeStressesAtGaussianPoints(elementDisplacements);
      avgStress.fill(0.);
      for (unsigned int sindex = 0; sindex < elementStresses.size(); sindex++) {
        avgStress += elementStresses[sindex];
      }
      avgStress /= double(elementStresses.size());
      allElementAvgStresses[elementIndex] = avgStress;
    }
    return allElementAvgStresses;    
  }


  vector<Vector3d>
  computeElementStrains(const vector<Vector2d> & displacements) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<Vector3d> allElementAvgStrains(_elements.size());

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacements;
    vector<Vector3d> elementStrains;
    Vector3d avgStrain; avgStrain.fill(0.);

    #pragma omp parallel for schedule(static) private(elementNodeIds, elementDisplacements, elementStrains, avgStrain)
    for (size_t elementIndex = _elementOffsetSize[rank]; elementIndex < _elementOffsetSize[rank]+_elementChunkSize[rank]; elementIndex++) { 
      elementNodeIds = _elements[elementIndex].getNodeIds();
      elementDisplacements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, displacements);
      elementStrains = _elements[elementIndex].computeStrainsAtGaussianPoints(elementDisplacements);
      avgStrain.fill(0.);
      for (unsigned int sindex = 0; sindex < elementStrains.size(); sindex++) {
        avgStrain += elementStrains[sindex];
      }
      avgStrain /= double(elementStrains.size());
      allElementAvgStrains[elementIndex] = avgStrain;
    }
    return allElementAvgStrains;
  }


  
  vector<double>
  computeElementEnergyDensities(const vector<Vector2d> & displacements) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<double> allElementAvgEnergyDensities(_elements.size());
    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacements;
    vector<double> elementEnergyDensities;
    double avgEnergyDensity = 0.;

    #pragma omp parallel for schedule(static) private(elementNodeIds,elementDisplacements,elementEnergyDensities,avgEnergyDensity)
    //for (unsigned int elementIndex = 0; elementIndex < _elements.size(); elementIndex++) {
    for (size_t elementIndex = _elementOffsetSize[rank]; elementIndex < _elementOffsetSize[rank]+_elementChunkSize[rank]; elementIndex++) { 
      elementNodeIds = _elements[elementIndex].getNodeIds();
      elementDisplacements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, displacements);
      elementEnergyDensities = _elements[elementIndex].computeEnergyDensityAtGaussianPoints(elementDisplacements);
      avgEnergyDensity = 0.;
      for (unsigned int sindex = 0; sindex < elementEnergyDensities.size(); sindex++) {
        avgEnergyDensity += elementEnergyDensities[sindex];
      }
      avgEnergyDensity /= double(elementEnergyDensities.size());
      allElementAvgEnergyDensities[elementIndex] = avgEnergyDensity;
    }
    return allElementAvgEnergyDensities;    
  }
  

  vector<vector<Vector3d> >
  computeWorldStrainsAtGaussianPoints(const vector<Vector2d> & displacements) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<vector<Vector3d> > allStrainsAtGaussianPoints(_elements.size());
    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacements;

    #pragma omp parallel for schedule(static) private(elementNodeIds, elementDisplacements)
    for (size_t elementIndex = _elementOffsetSize[rank]; elementIndex < _elementOffsetSize[rank]+_elementChunkSize[rank]; elementIndex++) { 
      elementNodeIds = _elements[elementIndex].getNodeIds();
      elementDisplacements = Utilities::getElementDisplacementsFromGlobalList(elementNodeIds, displacements);
      allStrainsAtGaussianPoints[elementIndex] = _elements[elementIndex].computeStrainsAtGaussianPoints(elementDisplacements); 
    }
    return allStrainsAtGaussianPoints;
  }

  vector<vector<double> >
  computeWorldPhisAtGaussianPoints(const vector<double> & phis) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<vector<double> > allPhisAtGaussianPoints(_elements.size());
    vector<size_t> elementNodeIds;
    vector<double> elementPhis;

    #pragma omp parallel for schedule(static) private(elementNodeIds, elementPhis)
    for (size_t elementIndex = _elementOffsetSize[rank]; elementIndex < _elementOffsetSize[rank]+_elementChunkSize[rank]; elementIndex++) { 
      elementNodeIds = _elements[elementIndex].getNodeIds();
      elementPhis = Utilities::getElementPhisFromGlobalList(elementNodeIds, phis);
      allPhisAtGaussianPoints[elementIndex] = _elements[elementIndex].computePhisAtGaussianPoints(elementPhis); 
    }  
    return allPhisAtGaussianPoints;
  }

  void
  computeWorldNodalStresses(const vector<Vector2d> & displacements,
    vector<double> & nodalStressFirstComponents, vector<double> & nodalStressSecondComponents,
    vector<double> & nodalStressThirdComponents) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const size_t globalNodeSize = displacements.size();
         
    nodalStressFirstComponents.resize(globalNodeSize);   
    nodalStressSecondComponents.resize(globalNodeSize);  
    nodalStressThirdComponents.resize(globalNodeSize);


    vector<double> nodalVolumeSums(globalNodeSize);
    vector<double> nodalVolumeSumsPerTask(globalNodeSize);              
    vector<double> nodalStressFirstComponentsPerTask(globalNodeSize);   
    vector<double> nodalStressSecondComponentsPerTask(globalNodeSize);  
    vector<double> nodalStressThirdComponentsPerTask(globalNodeSize);    

    #pragma omp parallel for schedule(static)
    for (size_t nodeIndex = 0; nodeIndex < globalNodeSize; nodeIndex++) {
      nodalVolumeSumsPerTask[nodeIndex] = 0.;
      nodalStressFirstComponentsPerTask[nodeIndex] = 0.;
      nodalStressSecondComponentsPerTask[nodeIndex] = 0.;
      nodalStressThirdComponentsPerTask[nodeIndex] = 0.;
    }

    vector<size_t>      elementNodeIds;
    vector<double>      elementWeights;
    vector<Vector3d>    elementStressesPerTask = computeElementStresses(displacements);

    #pragma omp parallel private(elementNodeIds,elementWeights)
    {
      vector<double> threadVolumeSums = nodalVolumeSumsPerTask;
      vector<double> threadNodalStressFirstComponents = nodalStressFirstComponentsPerTask;
      vector<double> threadNodalStressSecondComponents = nodalStressSecondComponentsPerTask;
      vector<double> threadNodalStressThirdComponents = nodalStressThirdComponentsPerTask;
      #pragma omp for schedule(static)
      for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        elementNodeIds = _elements[eindex].getNodeIds();
        elementWeights = _elements[eindex].getNodalWeights();
        for (size_t nindex = 0; nindex < elementNodeIds.size(); nindex++) {
          threadNodalStressFirstComponents[elementNodeIds[nindex]] += elementStressesPerTask[eindex](0)*elementWeights[nindex];
          threadNodalStressSecondComponents[elementNodeIds[nindex]] += elementStressesPerTask[eindex](1)*elementWeights[nindex];
          threadNodalStressThirdComponents[elementNodeIds[nindex]] += elementStressesPerTask[eindex](2)*elementWeights[nindex];
          threadVolumeSums[elementNodeIds[nindex]] += elementWeights[nindex];
        }
      }
      #pragma omp critical
      {
        for (size_t index = 0; index < globalNodeSize; index++) {
          nodalVolumeSumsPerTask[index] += threadVolumeSums[index];
          nodalStressFirstComponentsPerTask[index] += threadNodalStressFirstComponents[index];
          nodalStressSecondComponentsPerTask[index] += threadNodalStressSecondComponents[index];
          nodalStressThirdComponentsPerTask[index] += threadNodalStressThirdComponents[index];
        }
      }
    }

    MPI_Allreduce(nodalVolumeSumsPerTask.data(), nodalVolumeSums.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t nindex = 0; nindex < globalNodeSize; nindex++) {
      nodalStressFirstComponentsPerTask[nindex] /= nodalVolumeSums[nindex];
      nodalStressSecondComponentsPerTask[nindex] /= nodalVolumeSums[nindex];
      nodalStressThirdComponentsPerTask[nindex] /= nodalVolumeSums[nindex];
    }

    MPI_Allreduce(nodalStressFirstComponentsPerTask.data(), nodalStressFirstComponents.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(nodalStressSecondComponentsPerTask.data(), nodalStressSecondComponents.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(nodalStressThirdComponentsPerTask.data(), nodalStressThirdComponents.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }


  void
  computeWorldNodalStrains(const vector<Vector2d> & displacements,
    vector<double> & nodalStrainFirstComponents, vector<double> & nodalStrainSecondComponents,
    vector<double> & nodalStrainThirdComponents) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const size_t globalNodeSize = displacements.size();
            
    nodalStrainFirstComponents.resize(globalNodeSize);    
    nodalStrainSecondComponents.resize(globalNodeSize);   
    nodalStrainThirdComponents.resize(globalNodeSize);


    vector<double> nodalVolumeSums(globalNodeSize);
    vector<double> nodalVolumeSumsPerTask(globalNodeSize);               
    vector<double> nodalStrainFirstComponentsPerTask(globalNodeSize);    
    vector<double> nodalStrainSecondComponentsPerTask(globalNodeSize);   
    vector<double> nodalStrainThirdComponentsPerTask(globalNodeSize);

    #pragma omp parallel for schedule(static)
    for (size_t nodeIndex = 0; nodeIndex < globalNodeSize; nodeIndex++) {
      nodalVolumeSumsPerTask[nodeIndex] = 0.;
      nodalStrainFirstComponentsPerTask[nodeIndex] = 0.;
      nodalStrainSecondComponentsPerTask[nodeIndex] = 0.;
      nodalStrainThirdComponentsPerTask[nodeIndex] = 0.;
    } 

    vector<size_t>      elementNodeIds;
    vector<double>      elementWeights;
    vector<Vector3d>    elementStrainsPerTask = computeElementStrains(displacements);


    #pragma omp parallel private(elementNodeIds, elementWeights)
    {
      vector<double> threadVolumeSums = nodalVolumeSumsPerTask;
      vector<double> threadNodalStrainFirstComponents = nodalStrainFirstComponentsPerTask;
      vector<double> threadNodalStrainSecondComponents = nodalStrainSecondComponentsPerTask;
      vector<double> threadNodalStrainThirdComponents = nodalStrainThirdComponentsPerTask;
      #pragma omp for schedule(static)
      for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        elementNodeIds = _elements[eindex].getNodeIds();
        elementWeights = _elements[eindex].getNodalWeights();
        for (size_t nindex = 0; nindex < elementNodeIds.size(); nindex++) {
          threadNodalStrainFirstComponents[elementNodeIds[nindex]] += elementStrainsPerTask[eindex](0)*elementWeights[nindex];
          threadNodalStrainSecondComponents[elementNodeIds[nindex]] += elementStrainsPerTask[eindex](1)*elementWeights[nindex];
          threadNodalStrainThirdComponents[elementNodeIds[nindex]] += elementStrainsPerTask[eindex](2)*elementWeights[nindex];          
          threadVolumeSums[elementNodeIds[nindex]] += elementWeights[nindex];
        }
      }
      #pragma omp critical
      {
        for (size_t index = 0; index < globalNodeSize; index++) {
          nodalVolumeSumsPerTask[index] += threadVolumeSums[index];
          nodalStrainFirstComponentsPerTask[index] += threadNodalStrainFirstComponents[index];
          nodalStrainSecondComponentsPerTask[index] += threadNodalStrainSecondComponents[index];
          nodalStrainThirdComponentsPerTask[index] += threadNodalStrainThirdComponents[index];
        }
      }
    }

    MPI_Allreduce(nodalVolumeSumsPerTask.data(), nodalVolumeSums.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t nindex = 0; nindex < globalNodeSize; nindex++) {
      nodalStrainFirstComponentsPerTask[nindex] /= nodalVolumeSums[nindex];
      nodalStrainSecondComponentsPerTask[nindex] /= nodalVolumeSums[nindex];
      nodalStrainThirdComponentsPerTask[nindex] /= nodalVolumeSums[nindex];
    }
    MPI_Allreduce(nodalStrainFirstComponentsPerTask.data(), nodalStrainFirstComponents.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(nodalStrainSecondComponentsPerTask.data(), nodalStrainSecondComponents.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(nodalStrainThirdComponentsPerTask.data(), nodalStrainThirdComponents.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }


  void
  computeWorldNodalEnergyDensities(const vector<Vector2d> & displacements, vector<double> & nodalEnergyDensities) const {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const size_t globalNodeSize = displacements.size();

    nodalEnergyDensities.resize(globalNodeSize);

    vector<double> nodalVolumeSums(globalNodeSize);

    vector<double> nodalVolumeSumsPerTask(globalNodeSize);
    vector<double> nodalEnergyPerTask(globalNodeSize);

    #pragma omp parallel for schedule(static)
    for (size_t nindex = 0; nindex < globalNodeSize; nindex++) {
      nodalVolumeSumsPerTask[nindex] = 0.;
      nodalEnergyPerTask[nindex] = 0.;
    }

    vector<size_t> elementNodeIds;
    vector<double> elementWeights;
    vector<double> elementEnergyDensities = computeElementEnergyDensities(displacements);

    #pragma omp parallel private(elementNodeIds, elementWeights)
    { 
      vector<double> threadNodalEnergy = nodalEnergyPerTask;
      vector<double> threadVolumeSums = nodalVolumeSumsPerTask;
      #pragma omp for schedule(static)
      for (size_t eindex = _elementOffsetSize[rank]; eindex < _elementOffsetSize[rank]+_elementChunkSize[rank]; eindex++) {
        elementNodeIds = _elements[eindex].getNodeIds();
        elementWeights = _elements[eindex].getNodalWeights();
        for (unsigned int nindex = 0; nindex < elementNodeIds.size(); nindex++) {
          threadNodalEnergy[elementNodeIds[nindex]] += elementEnergyDensities[eindex]*elementWeights[nindex];
          threadVolumeSums[elementNodeIds[nindex]] += elementWeights[nindex];
        }
      }
      #pragma omp critical
      {
        for (size_t index = 0; index < globalNodeSize; index++) {
          nodalEnergyPerTask[index] += threadNodalEnergy[index];
          nodalVolumeSumsPerTask[index] += threadVolumeSums[index];
        }
      }
    }
    MPI_Allreduce(nodalVolumeSumsPerTask.data(), nodalVolumeSums.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t nindex = 0; nindex < globalNodeSize; nindex++) {
      nodalEnergyPerTask[nindex] /= nodalVolumeSums[nindex];
    }
    MPI_Allreduce(nodalEnergyPerTask.data(), nodalEnergyDensities.data(), globalNodeSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }  


  void
  updatWorldNodePositionsVelocitiesAccelerations (const vector<Vector2d> & displacements, 
    const double & beta, const double & gamma, const double & dt) {

    vector<size_t> elementNodeIds;
    vector<Vector2d> elementDisplacements;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // update nodal points that ONLY considered in the current rank
    Vector2d currentVelocity;
    Vector2d currentAcceleration;
    #pragma omp parallel for private(currentVelocity, currentAcceleration) schedule(static)    
    for (size_t nindex = _phiStartingRowIndex[rank]; nindex < _phiStartingRowIndex[rank]+_phiNumberOfRow[rank]; nindex++) {
      currentVelocity = _nodeVelocities[nindex];
      currentAcceleration = _nodeAccelerations[nindex];
      _nodePositions[nindex] += displacements[nindex];
      _nodeVelocities[nindex] = (1-gamma/beta)*currentVelocity+gamma/beta/dt*displacements[nindex]-dt*(gamma/2./beta-1.)*currentAcceleration;
      _nodeAccelerations[nindex] = 1./beta/dt/dt*(displacements[nindex]-dt*currentVelocity)-(1.-2.*beta)/2./beta*currentAcceleration;      
    }
    
  }

  void
  updateWorldNodePhis (const vector<double> & phiIncrements) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel for schedule(static)    
    for (size_t nindex = _phiStartingRowIndex[rank]; nindex < _phiStartingRowIndex[rank]+_phiNumberOfRow[rank]; nindex++) {
      _nodePhis[nindex] += phiIncrements[nindex];
    }
    
  }

  vector<Vector2d>
  getWorldNodePositions() const {
    return _nodePositions;
  }

  vector<Vector2d>
  getWorldNodeVelocities() const {
    return _nodeVelocities;
  }

  vector<Vector2d>
  getWorldNodeAccelerations() const {
    return _nodeAccelerations;
  }

  vector<double>
  getWorldNodePhis() const {
    return _nodePhis;
  }

  vector<SimplexTriangle>
  getWorldElements() const {
    return _elements;
  }

  SimplexTriangle
  getOneElement(const size_t & id) {
    return _elements[id];
  }  

  size_t
  getWorldNumberOfNodes() const {
    return _numberOfNodes;
  }

  size_t
  getWorldNumberOfElements() const {
    return _numberOfElements;
  }

  vector<vector<double> >
  getWorldStrainHistoryField() const {
    vector< vector<double> > strainHistoryField(_elements.size());
    for (unsigned int eindex = 0; eindex < _elements.size(); eindex++) {
      for (unsigned qpindex = 0; qpindex < _elements[eindex].getElementStrainHistory().size(); qpindex++) {
        strainHistoryField[eindex] = _elements[eindex].getElementStrainHistory();
      }
    }
    return strainHistoryField;
  }

  void
  getWorldMassKinematicVector(const double & beta, const double & dt, VectorXd & worldMassForce) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int indexChunkSize[_nodeBasedDisplacementChunkSize.size()];
    int indexOffsetSize[_nodeBasedDisplacementStartingIndex.size()];
    copy(_nodeBasedDisplacementChunkSize.begin(), _nodeBasedDisplacementChunkSize.end(), indexChunkSize);
    copy(_nodeBasedDisplacementStartingIndex.begin(), _nodeBasedDisplacementStartingIndex.end(), indexOffsetSize);

    worldMassForce.resize(_numberOfNodes*2);
    VectorXd worldMassForcePerTask(_phiNumberOfRow[rank]*2); 
    worldMassForcePerTask.fill(0.);

    size_t localRowIndex = 0;

    #pragma omp parallel for schedule(static) private(localRowIndex)
    for (size_t nindex = _phiStartingRowIndex[rank]; nindex < _phiStartingRowIndex[rank]+_phiNumberOfRow[rank]; nindex++) {
      localRowIndex = nindex -_phiStartingRowIndex[rank];
      worldMassForcePerTask(localRowIndex*2)   = _nodeVelocities[nindex](0)/beta/dt+(1/2./beta-1.)*_nodeAccelerations[nindex](0);
      worldMassForcePerTask(localRowIndex*2+1) = _nodeVelocities[nindex](1)/beta/dt+(1/2./beta-1.)*_nodeAccelerations[nindex](1);
    }
    MPI_Allgatherv(worldMassForcePerTask.data(), indexChunkSize[rank], MPI_DOUBLE, worldMassForce.data(), indexChunkSize, indexOffsetSize, MPI_DOUBLE, MPI_COMM_WORLD);   
  }


  void
  getWorldDampingKinematicVector(const double & beta, const double & gamma, const double & dt, VectorXd & worldDampingForce) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int indexChunkSize[_nodeBasedDisplacementChunkSize.size()];
    int indexOffsetSize[_nodeBasedDisplacementStartingIndex.size()];
    copy(_nodeBasedDisplacementChunkSize.begin(), _nodeBasedDisplacementChunkSize.end(), indexChunkSize);
    copy(_nodeBasedDisplacementStartingIndex.begin(), _nodeBasedDisplacementStartingIndex.end(), indexOffsetSize);

    worldDampingForce.resize(_numberOfNodes*2);
    VectorXd worldDampingForcePerTask(_phiNumberOfRow[rank]*2);
    worldDampingForcePerTask.fill(0.);

    size_t localRowIndex = 0;

    #pragma omp parallel for schedule(static) private(localRowIndex)
    for (size_t nindex = _phiStartingRowIndex[rank]; nindex < _phiStartingRowIndex[rank]+_phiNumberOfRow[rank]; nindex++) {
      localRowIndex = nindex - _phiStartingRowIndex[rank];
      worldDampingForcePerTask(localRowIndex*2) = (gamma/beta-1)*_nodeVelocities[nindex](0)+dt*(gamma/2./beta-1.)*_nodeAccelerations[nindex](0);
      worldDampingForcePerTask(localRowIndex*2+1) = (gamma/beta-1)*_nodeVelocities[nindex](1)+dt*(gamma/2./beta-1.)*_nodeAccelerations[nindex](1);
    }    
    MPI_Allgatherv(worldDampingForcePerTask.data(), indexChunkSize[rank], MPI_DOUBLE, worldDampingForce.data(), indexChunkSize, indexOffsetSize, MPI_DOUBLE, MPI_COMM_WORLD);  
  }

  double
  getK() const {
    return _k;
  }

  vector<size_t>
  getElementChunkSize() const {
    return _elementChunkSize;
  }

  vector<size_t>
  getElementOffsetSize() const {
    return _elementOffsetSize;
  }

  vector<int>
  getDisplacementStartingRowIndices() const {
    return _displacementStartingRowIndex;
  }

  vector<int>
  getDisplacementNumberOfRows() const {
    return _displacementNumberOfRow;
  }

  vector<int>
  getPhiStartingRowIndices() const {
    return _phiStartingRowIndex;
  }

  vector<int>
  getPhiNumberOfRows() const {
    return _phiNumberOfRow;
  }

  vector<int>
  getNodeBasedDisplacementStartingIndex() const {
    return _nodeBasedDisplacementStartingIndex;
  }

  vector<int>
  getNodeBasedDisplacementChunkSize() const {
    return _nodeBasedDisplacementChunkSize;
  }


private:
  vector<SimplexTriangle>     _elements;
  size_t                      _numberOfNodes;
  size_t                      _numberOfElements;
  vector<Vector2d>            _nodePositions;                              // all node positions
  vector<Vector2d>            _nodeVelocities;                             // all node velocities
  vector<Vector2d>            _nodeAccelerations;                          // all node accelerations
  double                      _k;
  vector<double>              _nodePhis;                                   // all node phis

  vector<size_t>              _elementChunkSize;
  vector<size_t>              _elementOffsetSize;


  vector<int>                 _displacementMatrixChunkSize;
  vector<int>                 _displacementMatrixOffsetSize;
  size_t                      _totalDisplacementMatrixNumber;

  vector<int>                 _phiMatrixChunkSize;
  vector<int>                 _phiMatrixOffsetSize;
  size_t                      _totalPhiMatrixNumber;

  vector<int>                 _displacementStartingRowIndex;
  vector<int>                 _displacementNumberOfRow;

  vector<int>                 _phiStartingRowIndex;
  vector<int>                 _phiNumberOfRow;

  vector<int>                 _nodeBasedDisplacementStartingIndex;
  vector<int>                 _nodeBasedDisplacementChunkSize;
  

};
#endif  // ASSEMBLER_H
