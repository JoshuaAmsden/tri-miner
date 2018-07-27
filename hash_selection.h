/*
 * hash_selection.h
 *
 *  Created on: Jul 9, 2018
 *      Author: Tri Nguyen
 */

#ifndef HASH_SELECTION_H_
#define HASH_SELECTION_H_

#include "uint256.h"
#include <string.h>
#include <string>

extern int GetSelection(uint256 blockHash, int index);
extern bool isScrambleHash(uint256 blockHash);
extern uint256 scrambleHash(uint256 blockHash);
extern std::string getHashSelectionsString(uint256 hash);





#endif /* HASH_SELECTION_H_ */
