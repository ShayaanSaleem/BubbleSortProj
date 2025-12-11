# Algorithm Name
Bubble Student Height Sorter

## Demo video/gif/screenshot of test


https://github.com/user-attachments/assets/d94aa3aa-152a-4b01-8fd8-372101b18f52


## Problem Breakdown & Computational Thinking 

  The 4 pillars of computation from Lecture 3 of Week 1 can be applied to understand and breakdown my application of bubble search for my project. The first pillar i'll be looking at is decomposition, which is the breaking down of a complex task into smaller, more manageable steps. For this project, I decomposed the idea of teaching bubble sort by comparing student height into first collecting heights from the user through a GUI, validating and storing those heights as integers, running the bubble sort passes one at a time and updating the characters' positions and colours, and lastly generating a written explanation panel that describes the steps occuring for the user's understanding. 
  The second pillar of computation i'll use to solve this problem is pattern recognition. Pattern recognition is identifying trends and using them to make predictions or decisions. In this context, I realize bubble sort always repeats the same pattern of scanning neighbours from left to right, comparing each pair, swapping when the left value is larger, and then shrinking the unsorted portion of the list after each pass. This means I can use this pattern in my project's design to compare the people's height and progressively rank them from shortest to tallest.
  The third pillar ill be analyzing is abstraction, which is focusing only on the essential aspects of the problem and ignoring irrelevant details. This applies in this situation, as the only things that will matter is the people's height (in cm). Anything else about them is mostly irrelevant. 
  Lastly, the fourth pillar is algorithms. This is the process of developing a step by step plan for solving a problem. I will likely approach this project with an input, processing, output flow. The user will enter or randomize the characters' heights, and then the bubble sort algorithm will iterate through passes and comparisons whilst tracking indices, and after each step the program will output a new line up and explanation.

<img width="733" height="744" alt="image" src="https://github.com/user-attachments/assets/3132e17e-bf19-4e71-860d-944f417b4ec3" />

- Why bubble sort is a good option to display students sorting themselves by height

1. bubble sort works by comparing neighbours in a line, which is how students realistically would compare heights in that situation... by looking at their neighbour and seeing if they're taller or shorter than themself
2. the "bubbling" effect of this sorting algorithm is easy to visualize with people, especially in a height situation
3. bubble sorting is very easy to explain step-by-step, making it optimal for TEACHING, which seemed to be an integral part of this project

- why students sorting themselves by height is a good situation to exemplify bubble sort:

1. heights are a familiar, real-world metric that can be sorted
2. the line of students maps cleanly to a 1d list in code

## Steps to Run

1. Choose how many people to compare heights between
- use the slider at the top of the program labelled "Number of students in the line" to pick the amount of students you want to compare height between
- when you change the amount of people through the slider, the app resets with that many people compared.

2. Enter or randomize the heights
- below the program's 5 main buttons directly under it's main window, you'll see labelled inputs (P1, P2, P3, ...)
  ^ in those spaces, input the students' heights in centimetres manually
- OR, for testing purposes, you can click the "randomize heights" button to automatically fill all visible inputs with random realistic heights

3. Set the heights
- click the "Set heights" button
- if you manually inputted the heights, clicking "set heights" should actually process those values and apply them to the people on the visual
- if you randomized heights, you don't need to click "set height"

4. Step through Bubble Sort
- you can either click the "Next step" button to perform one Bubble Sort comparison at a time
- OR, click the "Run to end" button and the algorithm will continue automatically until it's complete

5. Read explanation panel
- the right side of the screen is dedicated to explaining how bubble sort works in this context, and what's happening in the current step in the algorithm
- also lists heights from left to right at each stage

6. Reset and try new data
- clicking the "reset" button stops the ongoing algorithm, clears all height boces, and returns characters to their "idle" state
- you can then change the number of students with the slider again (step 1), enter a new set of heights or randomize again (step 2), and generally just repeat the whole process.

## Hugging Face Link

https://huggingface.co/spaces/ShayaanSaleem/BubbleSortProj/tree/main

## Author & Acknowledgment
Shayaan Saleem
