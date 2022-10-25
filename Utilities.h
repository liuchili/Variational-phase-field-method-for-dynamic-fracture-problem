// -*- C++ -*-
#ifndef UTILITIES_H
#define UTILITIES_H

#include "Definitions.h"

namespace Utilities {

	vector<Vector2d>
	getElementDisplacementsFromGlobalList(const vector<size_t> & elementNodeIds,
		                                  const vector<Vector2d> & displacements) {
		vector<Vector2d> elementDisplacements(elementNodeIds.size());
		for (unsigned int index = 0; index < elementNodeIds.size(); index++) {
			elementDisplacements[index] = displacements[elementNodeIds[index]];
		}
	return elementDisplacements;
	}

    vector<double>
    getElementPhisFromGlobalList(const vector<size_t> & elementNodeIds, const vector<double> & phis) {
        vector<double> elementPhis(elementNodeIds.size());
        for (unsigned int index = 0; index < elementNodeIds.size(); index++) {
            elementPhis[index] = phis[elementNodeIds[index]];
        }
        return elementPhis;
    }

	vector<Vector2d>
	distributeGlobalVectorsToLocalVectors(const VectorXd & globalVector) {
		const unsigned int Dof = 2;
		unsigned int numberOfNodes = globalVector.size()/Dof;
		vector<Vector2d> localVectors(numberOfNodes);
		#pragma omp parallel for schedule(static)
		for (unsigned int nindex = 0; nindex < numberOfNodes; nindex++) {
			for (unsigned int coordinate = 0; coordinate < Dof; coordinate++) {
				localVectors[nindex](coordinate) = globalVector(nindex*Dof+coordinate);
			}
		}
		return localVectors; 
	}

    vector<double>
    distributeGlobalVectorsToLocalSacalars(const VectorXd & globalVector) {
        vector<double> localScalars(globalVector.size());
        #pragma omp parallel for schedule(static)
        for (unsigned int nindex = 0; nindex < localScalars.size(); nindex++) {
            localScalars[nindex] = globalVector(nindex);
        }
        return localScalars;
    }

	VectorXd
	convertLocalNodalDisplacementToGlobalList(const vector<Vector2d> & localDisplacements) {
		VectorXd globalDisplacements; globalDisplacements.resize(localDisplacements.size()*2);
		#pragma omp parallel for schedule(static)
		for (unsigned int nindex = 0; nindex < localDisplacements.size(); nindex++) {
			globalDisplacements(nindex*2) = localDisplacements[nindex](0);
			globalDisplacements(nindex*2+1) = localDisplacements[nindex](1);
		}
		return globalDisplacements;
	}

    VectorXd
    convertLocalNodalPhiToGlobalList(const vector<double> & localPhis) {
        VectorXd globalPhis; globalPhis.resize(localPhis.size());
        #pragma omp parallel for schedule(static)
        for (unsigned int nindex = 0; nindex < localPhis.size(); nindex++) {
            globalPhis(nindex) = localPhis[nindex];
        }
        return globalPhis;
    }

    VectorXd
    convertLocalElementalStrainHistoryToGlobalList(const vector< vector<double> > & localStrainHistories) {
        // assuming every element has the same number of gaussian points
        size_t subDim = localStrainHistories[0].size();
        VectorXd globalStrainHistories; globalStrainHistories.resize(localStrainHistories.size()*subDim);
        #pragma omp parallel for schedule(static)
        for (unsigned int index = 0; index < localStrainHistories.size(); index++) {
            for (unsigned int qpindex = 0; qpindex < subDim; qpindex++) {
                globalStrainHistories(index*subDim+qpindex) = localStrainHistories[index][qpindex];
            }
        }
        return globalStrainHistories;
    }
	
	Matrix<double, 2, 2>
	convertVoigtStressToMatrixForm (const Vector3d & voigtStress) {
		Matrix<double, 2, 2> stress; stress.fill(0.);
		stress(0,0) = voigtStress(0);
		stress(0,1) = voigtStress(2);
		stress(1,0) = voigtStress(2);
		stress(1,1) = voigtStress(1);
		return stress;
	}

	double
	Heaviside(const double & value) {
		if (value > 0) {
			return 1.;
		}
		else {
			return 0.;
		}
	}

	double
	SignCheck(const double & value) {
		if (value > 0) {
			return 1;
		}
		else if (value < 0) {
			return -1;
		}
		else {
			return 0;
		}
	}

	vector<size_t>
	returnMatrixIndexFromFourthOrderIndex (const size_t & i, const size_t & j, const size_t & k, const size_t & l) {
		size_t rowIndex = 0;
		size_t colIndex = 0;
		vector<size_t> Indices(2);
		if (i==j) {
			if (i==0) { //i=j=0
				if (k==0&&l==0) {
					rowIndex = 0;
					colIndex = 0;
				}
				else if (k==1&&l==1) {
					rowIndex = 0;
					colIndex = 1;
				}
				else { //k=0,l=1 or k=1,l=0
					rowIndex = 0;
					colIndex = 2;
				}
			}
			else if (i==1) { //i=j=1
				if (k==0&&l==0) {
					rowIndex = 1;
					colIndex = 0;
				}
				else if (k==1&&l==1) {
					rowIndex = 1;
					colIndex = 1;
				}
				else { //k=0,l=1 or k=1,l=0
					rowIndex = 1;
					colIndex = 2;
				}
			}
		}
		else if (i==0 && j==1){ 
			if (k==0&&l==0) {
				rowIndex = 2;
				colIndex = 0;
			}
			else if (k==1&&l==1) {
				rowIndex = 2;
				colIndex = 1;
			}
			else { //k=0,l=1 or k=1,l=0
				rowIndex = 2;
				colIndex = 2;
			}
		}
        else if (i==1 && j==0) {
            if (k==0&&l==0) {
                rowIndex = 2;
                colIndex = 0;
            }
            else if (k==1&&l==1) {
                rowIndex = 2;
                colIndex = 1;
            }
            else {
                rowIndex = 2;
                colIndex = 2;
            }
        }		
		Indices[0] = rowIndex;
		Indices[1] = colIndex;
		return Indices;
	}


	Matrix<double, 3, 3>
	getBarDMatrixWithoutCoefficient() {
		Matrix<double, 3, 3> dMatrix;
		dMatrix << 1., 1., 0,
		           1., 1., 0.,
		           0., 0., 0.;
		return dMatrix;
	}

	vector<Matrix<double, 3, 3> >
	getPlusMinusPMatrix(const vector<Vector2d> & principleStrainsAndVecs, const vector<Matrix<size_t, 6, 1> > & indexContainer) {
		vector<Matrix<double, 3, 3> > PMatrices(2); 
		Matrix<double, 3, 3> plusPMatrix; plusPMatrix.fill(0.);
		Matrix<double, 3, 3> minusPMatrix; minusPMatrix.fill(0.);
		const double delta = 1e-9;

        vector<double> principleStrain(2);
        vector<Vector2d> principleStrainVec(2);
        principleStrain[0] = principleStrainsAndVecs[0](0);
        principleStrain[1] = principleStrainsAndVecs[0](1);
        principleStrainVec[0] = principleStrainsAndVecs[1];
        principleStrainVec[1] = principleStrainsAndVecs[2];


		Vector2d perturbedStrain;
		perturbedStrain(0) = principleStrain[0];
		perturbedStrain(1) = principleStrain[1];

		if (perturbedStrain(0)-perturbedStrain(1)==0) {
			perturbedStrain(0) = perturbedStrain(0)*(1+delta);
			perturbedStrain(1) = perturbedStrain(1)*(1-delta);
		}

	    Vector2d positiveStrain; positiveStrain.fill(0.);
	    positiveStrain(0) = (perturbedStrain(0)+fabs(perturbedStrain(0)))/2;
	    positiveStrain(1) = (perturbedStrain(1)+fabs(perturbedStrain(1)))/2;

	    Vector2d negativeStrain; negativeStrain.fill(0.);
	    negativeStrain(0) = (perturbedStrain(0)-fabs(perturbedStrain(0)))/2;
	    negativeStrain(1) = (perturbedStrain(1)-fabs(perturbedStrain(1)))/2;

        /*
	    size_t vectorIndex = 0;
        vector<size_t> indices;
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < 2; j++) {
				for (size_t k = 0; k < 2; k++) {
					for (size_t l = 0; l < 2; l++) {
						vectorIndex = 8*i+4*j+2*k+l;
						indices = returnMatrixIndexFromFourthOrderIndex(i,j,k,l);
						//cout << vectorIndex << " " << i << " " << j << " " << k << " " << l << " " << indices[0] << " " << indices[1] << endl;

						plusPMatrix(indices[0],indices[1]) += Heaviside(perturbedStrain(0))*principleStrainVec[0](i)*principleStrainVec[0](j)*principleStrainVec[0](k)*principleStrainVec[0](l)+
						                                      Heaviside(perturbedStrain(1))*principleStrainVec[1](i)*principleStrainVec[1](j)*principleStrainVec[1](k)*principleStrainVec[1](l)+
						                                      0.5*(positiveStrain(0)-positiveStrain(1))/(perturbedStrain(0)-perturbedStrain(1))*principleStrainVec[0](i)*principleStrainVec[1](j)*
						                                      (principleStrainVec[0](k)*principleStrainVec[1](l)+principleStrainVec[1](k)*principleStrainVec[0](l))+
						                                      0.5*(positiveStrain(1)-positiveStrain(0))/(perturbedStrain(1)-perturbedStrain(0))*principleStrainVec[1](i)*principleStrainVec[0](j)*
						                                      (principleStrainVec[1](k)*principleStrainVec[0](l)+principleStrainVec[0](k)*principleStrainVec[1](l));

						minusPMatrix(indices[0],indices[1]) += Heaviside(-1.*perturbedStrain(0))*principleStrainVec[0](i)*principleStrainVec[0](j)*principleStrainVec[0](k)*principleStrainVec[0](l)+
						                                      Heaviside(-1.*perturbedStrain(1))*principleStrainVec[1](i)*principleStrainVec[1](j)*principleStrainVec[1](k)*principleStrainVec[1](l)+
						                                      0.5*(negativeStrain(0)-negativeStrain(1))/(perturbedStrain(0)-perturbedStrain(1))*principleStrainVec[0](i)*principleStrainVec[1](j)*
						                                      (principleStrainVec[0](k)*principleStrainVec[1](l)+principleStrainVec[1](k)*principleStrainVec[0](l))+
						                                      0.5*(negativeStrain(1)-negativeStrain(0))/(perturbedStrain(1)-perturbedStrain(0))*principleStrainVec[1](i)*principleStrainVec[0](j)*
						                                      (principleStrainVec[1](k)*principleStrainVec[0](l)+principleStrainVec[0](k)*principleStrainVec[1](l));						                                     
					}
				}
			}
		}
		*/
        	
        
        size_t i,j,k,l;
		for (size_t index = 0; index < indexContainer.size(); index++) {
			i = indexContainer[index](0,0);
			j = indexContainer[index](1,0);
			k = indexContainer[index](2,0);
			l = indexContainer[index](3,0);

			plusPMatrix(indexContainer[index](4,0),indexContainer[index](5,0)) += Heaviside(perturbedStrain(0))*principleStrainVec[0](i)*principleStrainVec[0](j)*principleStrainVec[0](k)*principleStrainVec[0](l)+
			                                      Heaviside(perturbedStrain(1))*principleStrainVec[1](i)*principleStrainVec[1](j)*principleStrainVec[1](k)*principleStrainVec[1](l)+
			                                      0.5*(positiveStrain(0)-positiveStrain(1))/(perturbedStrain(0)-perturbedStrain(1))*principleStrainVec[0](i)*principleStrainVec[1](j)*
			                                      (principleStrainVec[0](k)*principleStrainVec[1](l)+principleStrainVec[1](k)*principleStrainVec[0](l))+
			                                      0.5*(positiveStrain(1)-positiveStrain(0))/(perturbedStrain(1)-perturbedStrain(0))*principleStrainVec[1](i)*principleStrainVec[0](j)*
			                                      (principleStrainVec[1](k)*principleStrainVec[0](l)+principleStrainVec[0](k)*principleStrainVec[1](l));			

			minusPMatrix(indexContainer[index](4,0),indexContainer[index](5,0)) += Heaviside(-1.*perturbedStrain(0))*principleStrainVec[0](i)*principleStrainVec[0](j)*principleStrainVec[0](k)*principleStrainVec[0](l)+
			                                      Heaviside(-1.*perturbedStrain(1))*principleStrainVec[1](i)*principleStrainVec[1](j)*principleStrainVec[1](k)*principleStrainVec[1](l)+
			                                      0.5*(negativeStrain(0)-negativeStrain(1))/(perturbedStrain(0)-perturbedStrain(1))*principleStrainVec[0](i)*principleStrainVec[1](j)*
			                                      (principleStrainVec[0](k)*principleStrainVec[1](l)+principleStrainVec[1](k)*principleStrainVec[0](l))+
			                                      0.5*(negativeStrain(1)-negativeStrain(0))/(perturbedStrain(1)-perturbedStrain(0))*principleStrainVec[1](i)*principleStrainVec[0](j)*
			                                      (principleStrainVec[1](k)*principleStrainVec[0](l)+principleStrainVec[0](k)*principleStrainVec[1](l));
		}
		
		
        
        plusPMatrix(0,2) = plusPMatrix(0,2)*0.5;
        minusPMatrix(0,2) = minusPMatrix(0,2)*0.5;

        plusPMatrix(1,2) = plusPMatrix(1,2)*0.5;
        minusPMatrix(1,2) = minusPMatrix(1,2)*0.5;

        plusPMatrix(2,0) = plusPMatrix(2,0)*0.5;
        minusPMatrix(2,0) = minusPMatrix(2,0)*0.5;

        plusPMatrix(2,1) = plusPMatrix(2,1)*0.5;
        minusPMatrix(2,1) = minusPMatrix(2,1)*0.5;

        plusPMatrix(2,2) = plusPMatrix(2,2)*0.25;
        minusPMatrix(2,2) = minusPMatrix(2,2)*0.25;
        
        // symmetrize
        plusPMatrix = (plusPMatrix+plusPMatrix.transpose())/2.;
        minusPMatrix = (minusPMatrix+minusPMatrix.transpose())/2.;

        double avg = 0.;
        // making sure P+ and P-, when added together, has zero off-diagonal terms
	    for (unsigned int i = 0; i < 2; i++) {
	    	for (unsigned int j = i+1; j < 3; j++) {
	    		avg = (fabs(plusPMatrix(i,j))+fabs(minusPMatrix(i,j)))/2.;
	    		plusPMatrix(i,j) = avg*SignCheck(plusPMatrix(i,j));
	    		plusPMatrix(j,i) = plusPMatrix(i,j);
	    		minusPMatrix(i,j) = avg*SignCheck(minusPMatrix(i,j));;
	    		minusPMatrix(j,i) = minusPMatrix(i,j);
	    	}
	    }	    
	            
		PMatrices[0] = plusPMatrix;
		PMatrices[1] = minusPMatrix;
		return PMatrices;
	}

	vector<Vector2d>
    computePrincipleStrainsAndVecs (const Vector3d & voigtStrain) {
        Matrix<double, 2, 2> strainTensor;
        strainTensor.fill(0.);
        vector<Vector2d> principleStrainsAndVecs(3);
        strainTensor(0,0) = voigtStrain(0);
        strainTensor(0,1) = voigtStrain(2)/2.;
        strainTensor(1,0) = voigtStrain(2)/2.;
        strainTensor(1,1) = voigtStrain(1);
        EigenSolver<MatrixXd> es(strainTensor);
        principleStrainsAndVecs[0](0) = es.eigenvalues()[0].real();
        principleStrainsAndVecs[0](1) = es.eigenvalues()[1].real();
        principleStrainsAndVecs[1] = es.eigenvectors().col(0).real();
        principleStrainsAndVecs[2] = es.eigenvectors().col(1).real();
        return principleStrainsAndVecs;
    }
}

#endif  // UTILITIES_H
