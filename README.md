# Ternary_Contour_Vid
Makes a video (.gif) of a stitched series of ternary contour plots while a window slides through a series of 3-dimensional data. Contours are based on 'spatial'-density of datapoints in the domain of the ternary plot.

<!-- Options -->
## Options
 User can specify the following parameters when calling the function:
 - colour of the colour gradient, according to cmap available options (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
 - the number of contours.
 - the end-members of the plot, which must match the content of the input dataset (systems are: [Si+Al, Mg, Fe], [Si+Al, Mg+Fe, O], or [S, Ni, Fe]).
 - the width of the window (in units of number of datapoints).
 - the starting datapoint which to generate the first video frame from.
 - the number of steps of the window (a subset of the entire dataset can be plotted, given this parameter, the width of the window and the starting datapoint specified by the user).
 - the frames per second.
 - to plot the position of the window, either in units of datapoint index or in associated distance of each datapoint from a datum (in the case that the series of data is from a line-scan for example).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Example output -->
## Example output

<br />
<div align="center">

[![Product Name Screen Shot][product-screenshot]](https://example.com)
 
 </div>

This figure was generated from the following user-specified options:
 - dataset = [[A1, B1, C1], [A2, B2, C2], ...[An, Bn, Cn]] where A = Si+Al, B = Fe, and C = Mg at.%
 - WindDispPositionData =[Distance1, Distance2, ... Distance3] in nm.
 - ContLines = 'n'
 - NumLevels = 7
 - Colour = 'Blues'
 - type = 'silicate'
 - WindDispType = 'distance'
 - WindowWidth = 10
 - NumberOfSteps = 300
 - StartingIndex_FirstWindow = 0
 - fps = 10

 
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community what it is. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Dan Hallatt - daniel.hallatt@univ-lille.fr

Project Link: [https://github.com/DanHallatt/Ternary_Contour_Vid](https://github.com/Ternary_Contour_Vid)

Associated Institute Link: https://umet.univ-lille.fr/MTP/index.php?lang=fr

<p align="right">(<a href="#top">back to top</a>)</p>


[product-screenshot]: Images/ExampleOutput.gif
