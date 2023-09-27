import matplotlib.pyplot as plt
from amworkflow.api import amWorkflow as aw
from matplotlib.widgets import Button
# Create an empty list to store the clicked points
points = []
# Initialize point click flag
point_click_enabled = True
is_close = False

def toggle_point_click(event):
    global point_click_enabled
    point_click_enabled = not point_click_enabled

def create_polygon():
    # Check if there are enough points to create a polygon (minimum 3 points)
    if len(points) >= 2:
        wall = aw.geom.CreateWallByPoints(points,30,0,is_close=is_close)
        loops = []
        for lp in wall.result_loops:              
                coords = [wall.pnts.pts_index[i] for i in lp]
                loops.append(coords)
        return loops,wall.result_loops, wall.pnts.pts_index

def onclick(event):
    # Check if the click happened within the axes
    if not point_click_enabled:
        return
    if event.inaxes:
        # Get the x and y coordinates of the click
        x, y = event.xdata, event.ydata
        # Add the point to the list
        points.append([x, y])
        # Clear the previous plot and replot the points
        ax.clear()
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        arrow_props = dict(arrowstyle='-|>', color='black')
        for i in range(1, len(points)):
            prev_x, prev_y = points[i - 1]
            current_x, current_y = points[i]
            ax.plot([prev_x, current_x], [prev_y, current_y], marker='o', linestyle='-', color='black')
            mid_x = (prev_x + current_x) / 2
            mid_y = (prev_y + current_y) / 2
            ax.annotate('', xy=(current_x, current_y), xytext=(prev_x, prev_y),
                        arrowprops=arrow_props)
        if is_close and len(points) > 2:
            current_x, current_y = points[-1]
            init_x, init_y = points[0]
            ax.annotate('', xy=(init_x, init_y), xytext=(current_x, current_y),
                        arrowprops=arrow_props, label = 'Central Path')
        ax.scatter(*zip(*points), marker='o', label='Input Points', color='black')
        
        # Draw polygons based on the clicked points
        
        if len(points) > 1:
            polygons, indices,pt_coord = create_polygon()
            for i,poly in enumerate(polygons):
                print(poly)
                x = [point[0] for point in poly]
                y = [point[1] for point in poly]
                plt.plot(x + [x[0]], y + [y[0]], linestyle='-', marker='o')  # Connect the last point to the first point to close the polygon
            for lp in indices:
                for pt in lp:
                    xi, yi,_ = pt_coord[pt]
                    plt.annotate(f'{pt}', (xi, yi), fontsize=12, ha='right',color = 'gray')
        # Create a button to toggle point clicking
        # toggle_button_ax = plt.axes([0.7, 0.01, 0.2, 0.04])
        # toggle_button = Button(toggle_button_ax, 'Toggle Point Click', color='lightgoldenrodyellow')
        # toggle_button.on_clicked(toggle_point_click)
        plt.tight_layout()
        plt.legend()
        plt.draw()

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_aspect('equal')
# Connect the click event to the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

# Set the limits of the plot

# plt.autoscale(False)

# Show the plot
plt.show()
