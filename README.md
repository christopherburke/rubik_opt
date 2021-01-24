# rubik_opt
Rubik's Cube optimal move solver. A python/cython solution to unscramble the Rubik's Cube in the fewest number of moves.

# Motivation
This code was inspired by watching the Netflix show Speedcubers about the current champions and simultaneously I was learning more about graph traversal algorithms ("Grokking Algorithms" by Aditya Bhargava, such a good book for introducing algorithms before diving into dry textbooks). It seemed like a perfect way to learn graph traversal algorithms with an actual problem to solve, the Rubik's Cube. Also, ya know, 2020 & 2021 needed some things to distract my mind. I started with writing a solver for a single cube to go from where it was to where I wanted it to go without messing up previously placed cubes. I stumbled into this blog post written by [Benjamin Botto](https://medium.com/@benjamin.botto/implementing-an-optimal-rubiks-cube-solver-using-korf-s-algorithm-bf750b332cf9 "Optimal Rubik's Cube Solving"), and I fell down a rabbit hole. Along the way I learned about Breadth First Search, Depth First Search, speeding up python with cython, dabbled in writing c++ for the first time, lehmer code (such a cool thing), A* graph traversal using distance to end pattern database, and implementing shared memory for the multiprocessing step. The solution method described in Ben Botto's blog (BBB, hereafter) excellently details the work from [Richard Korf, 1997](https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf).

# Entering Scrambled Cube to Solve
The nomenclature for entering the colors of the scrambled cube and moves is non-standard as it is left over from when I started to program before consulting vast amount of information on solving Rubiks cube. Situate the cube such that the white center face faces you, orange center face on top, and blue center face on the right of the cube. Visualize an x,y,z coordinate grid with the origin located at the lower left corner closest to you. The blue cube face to the right is the positive x-axis direction, the yellow cube face on the back of the cube is the positive y-axis direction, and the top orange face is the positive z-axis in this coordinate system. Now going back to the coordinate origin, start numbering the cubes from left to right and front to back. Thus, the lower left corner cubie closest to you is number 1, the next edge cubie along the lower front face is number 2 (along the positive x-axis). In this numbering scheme, the bottom center red cubie is number 5, front white center cubie is 11, left green center cubie is 13, the center hidden cube that is not visible is cube number 14, right face blue center cubie is 15, back face yellow center cubie is 17, top orange center is number 23, and the last cubie is the upper right cubie furthest away from you given number 27. For a particular cubie the face/sticker is specified by the direction that a normal pointing away from the cube. All cubie faces on the right face have a outward pointing normal in the positive x-axis direction which are given the shortened code 'px'. Faces on the yellow back face are towards the positive y-axis (py), faces on the top orange face are the positive z-axis (pz). The face code is a two character number of the cube and a two character face/sticker normal direction. '01my' designates the face of the lower left corner cubie number 1 with face normal pointing along negative axis. The integer code for color to use for a cubie face/sticker is 
```
1 = orange, 2 = blue, 3 = yellow, 4 = green, 5 = white, 6 = red
```
Here is the dictionary one would use for the solved Rubik's cube
```python
    solvedfaces = {"01my":5, "01mz":6, "01mx":4, "02my":5, "02mz":6,"03my":5, "03px":2, "03mz":6,\
                 "04mx":4, "04mz":6,"05mz":6,"06px":2, "06mz":6,\
                 "07mx":4, "07py":3, "07mz":6,"08py":3, "08mz":6,"09px":2, "09py":3, "09mz":6,\
                 "10mx":4, "10my":5,"11my":5,"12px":2, "12my":5,\
                 "13mx":4,"15px":2,\
                 "16mx":4, "16py":3,"17py":3,"18px":2, "18py":3,\
                 "19mx":4, "19my":5, "19pz":1,"20my":5, "20pz":1,"21px":2, "21my":5, "21pz":1,\
                 "22mx":4, "22pz":1,"23pz":1,"24px":2, "24pz":1,\
                 "25mx":4, "25py":3, "25pz":1,"26py":3, "26pz":1,"27px":2, "27py":3, "27pz":1}
```
It will hopefully guide you into entering the scrambled cube faces into the begcubefaces dictionary to solve.

The characters I use to encode the moves are also non standard. I use a system where half turn/180 deg face turns are allowed. Here 
Front face clockwise is FC, front face counter-clockwise is FG, and front face half turn FH. These are more typically called, F,F', and F2, respectively. Rather than viewing the cube from each face for determining clockwise or counter-clockwise, I keep the rubik cube fixed, thus Back face clockwise move as viewed from the front I call BC, but this would be B' in the standard notation.  Back face counter-clockwise viewed from the front I call BG, but would be B in the standard notation. The right face turned such that lower cubie goes up RU, normally R, and right face turned such that the lower cubie goes down RD' == R. Left face up LU==L' and left face down LD==L. Upper face turned such that left most cubie moves right UR==U', and upper face turned such that left most cubie moves left UL=='U'. Same with lower/down face DR=D and DL=D'. On output the moves are given in my move notation and also the standard move notation.

# Multiprocessing / Parallelization Notes
Rather than let a single process explore the tree for a solution, to parallelize the search, I record the end states of the cube faces after the first two moves have been performed. Note, that there are 18 moves in total, so after two moves there should be 18x18=324 cube states, but some of the second moves are redundant (see BBB), so there are 255 cube states. An individual worker continues the search for a solution commencing from one of these 255 cube states independently until they reach the current max level limit or find a solution. It is essentially an embarrasingly parallel program where workers do not need to communicate with each other. I work on a system with 4 Intel Xeon CPU E5-2620 v4 @ 2.10GHz processors each with 8 cores. Using 30 cores with python's multiprocessing pool.map() methods I achieve a speed up by a factor of 15 over the single core. Except for the bookkeeping, having the pattern databases put into a read-only shared memory space such that all the workers have access without needing copies was the part that took some searching around. My implementation is modelled after this description by [Mianzhi Wang](https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html "On Sharing Large Arrays When Using Python's Multiprocessing"). Other solutions on stackoverflow that involve just putting the variables in module space did work for access without copies, but did not lead to any speed ups. That method lead to 100 times more interrupts and context switches per second according to [vmstat](https://access.redhat.com/solutions/1160343) indicating some really inefficient memory access issues and jobs switching cores rapidly. However, Wang's method for shared memory really worked really well. I really wonder how this would scale to doing the solving with CUDA GPUs. I may try that as this system has an NVIDIA GTX 1050 Ti (with 768 CUDA cores). There are ~3200 states after three moves. The bottleneck may be acessing the 550MB worth of pattern databases from GPU device memory which wont fit on shared memory closer to the threads.

# Profiling
In the current version, the tall pole is the lehmer coding. It is 3 times slower than performing the ~14 moves per node. The bookkeeping of the DFS stack runs at a similar time to the moves. The lehmer code implementation is using a linear algorithm, so any improvements are likely in implementation. May be hard to improve upon, but I frankly did not explore optimizing yet nearly as much as the DFS bookkeeping and performing moves. The other route for improvement is through parallelizing with more cores and exploring the GPU option. Initial exploration shows that it is scaling well with increasing cores, so that is encouraging. I did try my hand at using C++ to speed things up. I got an implementation working, but tall pole of using deque storage class maintaining the DFS stack lead to too many allocations and deallocations that bogged it down. The same thing occurs with python data array structures. They were always the time sink compared to the moves. I ended up doing away all the storage classes and went to a straight 2d buffer array where I just memcopy entries into and increment/decrement a pointer to the head of the stack and not bother with actually erasing or deallocating entries when they are pop'd off. The DFS, moves, and lehmer coding were all moved to cython. The only thing left in python now is converting the begcubefaces dictionary of cube faces to the cube number and orientation used internally as well as setting up the pool of workers. However, you will find code to perform the DFS, moves, and lehmer coding on the python side as well. My workflow was development and testing in native python to make sure it was correct. Then profile and move things into cython during optimization. So you will see very similar code on the python side to maybe help see what the cython is doing if you're not too familar with cython. 

