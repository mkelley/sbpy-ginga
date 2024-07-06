# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A Ginga local plugin for measuring small solar system object astrometry.


Based on Ginga's Pick tool.

"""

__all__ = ["Astrometry"]

from typing import Tuple
import numpy as np
from ginga.GingaPlugin import LocalPlugin
from ginga.gw import Widgets
from ginga import ImageView
from ginga.canvas.CanvasObject import CanvasObjectBase
from ginga.canvas.types.layer import DrawingCanvas
from ginga.canvas.types.basic import Point, Text
from ginga.util import wcs
from sbpy.utils import optional_packages

try:
    from photutils.centroids import centroid_2dg
except ImportError:
    centroid_2dg = None


class CenteringRegion:
    """Image region for centering on objects.


    Parameters
    ----------
    shape : `~ginga.canvas.CanvasObject.CanvasObjectBase`
        Base the region on this canvas object.

    image_view : `~ginga.ImageView`
        Ginga image view on which the centering should be based.

    canvas : `~ginga.canvas.CanvasObject.CanvasObjectBase`
        The canvas onto which the region should be drawn.

    label : str, optional
        Label region with this string.

    """

    region_default_width: float = 7
    region_default_height: float = 7

    def __init__(
        self,
        shape: CanvasObjectBase,
        image_view: ImageView,
        canvas: DrawingCanvas,
        label: str = "Astrometry",
    ):
        self.image_view: ImageView = image_view
        self.canvas: DrawingCanvas = canvas

        # shape location on image
        x1, y1, x2, y2 = shape.get_llur()
        x, y = shape.get_center_pt()

        CompoundObject = self.canvas.get_draw_class("compoundobject")
        Point = self.canvas.get_draw_class("point")
        Text = self.canvas.get_draw_class("text")
        self.canvas_object = CompoundObject(
            shape,
            Point(x, y, 6, color="red"),
            Text(x1, y2, label, color=shape.color),
        )
        self.canvas.add(self.canvas_object)

        self.data: np.ma.MaskedArray
        self.update_image_data()

    def __del__(self):
        if self.canvas.has_object(self.canvas_object):
            self.canvas.delete_object(self.canvas_object)
        self.canvas.update_canvas()

    @classmethod
    def at_location(
        cls,
        x: float,
        y: float,
        image_view: ImageView,
        canvas: DrawingCanvas,
        label: str = "Astrometry",
        color: str = "cyan",
        width: float = region_default_width,
        height: float = region_default_height,
    ):
        """Create a box region at the given location.


        Parameters
        ----------
        x, y : float
            The center of the region.

        image_view : `~ginga.ImageView`
            Ginga image view on which the centering should be based.

        canvas : `~ginga.canvas.CanvasObject.CanvasObjectBase`
            The canvas onto which the region should be drawn.

        label : str, optional
            Label region with this string.

        color : str, optional
            The color of the region.

        width, height : float
            The region dimensions.

        """
        Box: object = canvas.get_draw_class("box")
        tag = canvas.add(Box(x, y, width // 2, height // 2, color=color))
        shape: CanvasObjectBase = canvas.get_object_by_tag(tag)
        canvas.delete_object_by_tag(tag)

        return CenteringRegion(shape, image_view, canvas, label=label)

    @property
    def shape(self) -> CanvasObjectBase:
        return self.canvas_object.objects[0]

    @property
    def peak(self) -> Point:
        return self.canvas_object.objects[1]

    @property
    def text(self) -> Text:
        return self.canvas_object.objects[2]

    def get_image_data(self) -> np.ma.MaskedArray:
        """Returns a cutout of the image data, masked by the region shape."""

        image = self.image_view.get_vip()
        return image.cutout_shape(self.shape)

    def update_image_data(self) -> None:
        """Update the internal copy of the image data."""
        self.data = self.get_image_data()

    def get_center_point(self) -> Tuple[float]:
        """Get the region's center point.


        Returns
        -------
        x, y : float

        """

        return self.shape.get_center_pt()

    def set_center_point(self, x: float, y: float) -> None:
        """Move the region to this center point."""

        self.shape.move_to_pt((x, y))
        self.update_image_data()
        self.canvas.update_canvas()

    def get_peak_point(self) -> Tuple[float, float]:
        """Get the peak marker's coordinates."""

        return self.peak.get_center_pt()

    def set_peak_point(self, x: float, y: float) -> None:
        """Set the peak marker's coordinates."""
        self.peak.move_to_pt((x, y))
        self.canvas.update_canvas()

    def get_peak_value(self) -> float:
        """Image value at the center pixel."""

        x1: int
        y1: int
        x1, y1 = self.get_llur()[:2]

        x: float
        y: float
        x, y = self.get_peak_point()

        x -= x1
        y -= y1

        return float(self.data[int(round(y)), int(round(x))])

    def get_llur(self) -> Tuple[float]:
        """Get the region's lower-left and upper-right coordinates.


        Returns
        -------
        x1, y1, x2, y2 : int
            The shape's bounding box rounded to the nearest pixel.

        """

        # this matches image corner calculation in
        # ginga.util.vip.ViewerImageProxy.get_shape_view
        return [int(round(x)) for x in self.shape.get_llur()]

    def centroid(self, method: str) -> Tuple[float, float]:
        """Centroid on the region center.

        The peak marker is not moved.


        Parameters
        ----------
        method : str
            Centroid method: none, peak, 2D Gaussian.


        Returns
        -------
        x, y : float

        """

        x1: int
        y1: int
        x1, y1 = self.get_llur()[:2]

        x: float
        y: float
        if method == "none":
            x, y = self.get_center_point()
        elif method == "peak":
            y, x = np.unravel_index(self.data.argmax(), self.data.shape)
            x += x1
            y += y1
        elif method == "2D Gaussian":
            x, y = centroid_2dg(self.data)
            x += x1
            y += y1
        else:
            raise RuntimeError(f"{method} is not a valid centering method")

        return float(x), float(y)


class Astrometry(LocalPlugin):
    """Ginga plugin for interactive cometary image enhancements.

    Some code based on Ginga's Pick tool.

    """

    def __init__(self, fv, fitsimage):
        """
        This method is called when the plugin is loaded for the  first
        time.  ``fv`` is a reference to the Ginga (reference viewer) shell
        and ``fitsimage`` is a reference to the specific ImageViewCanvas
        object associated with the channel on which the plugin is being
        invoked.
        You need to call the superclass initializer and then do any local
        initialization.
        """

        super(Astrometry, self).__init__(fv, fitsimage)

        self.layer_tag = "sbpy-astrometry-canvas"
        self.region = None  # was pick_obj
        self._textlabel = "Astrometry"

        # types of shapes that can be drawn for object regions
        self.region_types = [
            "box",
            "squarebox",
            "rectangle",
            "circle",
            "ellipse",
            "freepolygon",
            "polygon",
        ]

        # available centroid methods
        self.centering_methods = [
            "none",
            "peak",
        ]
        if optional_packages(
            "photutils", message="some centroiding options are disabled"
        ):
            self.centering_methods.extend(["2D Gaussian"])

        # get preferences
        prefs = self.fv.get_preferences()
        self.settings = prefs.create_category("plugin_sbpy_Astrometry")
        self.settings.load(onError="silent")

        self.sync_preferences()

        self.region_width = CenteringRegion.region_default_width
        self.region_height = CenteringRegion.region_default_height

        # Initialize metadata
        self.target = ""
        self.observer_location = ""

        # drawing canvas for image annotations
        self.draw_classes = self.fv.get_draw_classes()
        canvas = self.draw_classes.DrawingCanvas()
        canvas.enable_draw(True)
        canvas.enable_edit(True)
        canvas.set_drawtype(self.region_type, color="cyan", linestyle="dash")
        canvas.set_callback("draw-event", self.draw_callback)
        canvas.set_callback("edit-event", self.edit_callback)
        canvas.add_draw_mode(
            "move",
            down=self.button_down,
            move=self.button_drag,
            up=self.button_up,
        )
        canvas.register_for_cursor_drawing(self.fitsimage)
        canvas.set_surface(self.fitsimage)
        canvas.set_draw_mode("move")
        self.canvas = canvas

    def sync_preferences(self):
        """Sync preferences from settings to self.

        Parameters are typically defined here.

        """

        self.region_color = self.settings.get("region_color", "green")

        self.region_type = self.settings.get("region_type", "box")
        if self.region_type not in self.region_types:
            self.region_type = "box"

        self.max_region_size = self.settings.get("max_region_size", 1024)

        self.centering_method = self.settings.get("centering_method", None)
        if self.centering_method not in self.centering_methods:
            self.centering_method = (
                "2D Gaussian" if "2D Gaussian" in self.centering_methods else "peak"
            )

    def create_change_value_callback(self, attribute_name, type=str):
        """Create a basic callback that updates a value and associated label."""

        def callback(widget, value):
            setattr(self, attribute_name, type(value))
            getattr(self.w, "label_" + attribute_name).set_text(
                str(getattr(self, attribute_name))
            )
            return True

        return callback

    def create_entry_callback(self, attribute_name, type=str):
        """Create a basic entry callback that updates a value and associated label."""

        default_value = ""
        if type is int:
            default_value = 0
        elif type is float:
            default_value = 0.0

        def callback(widget):
            value = default_value
            widget_value = widget.get_text().strip()
            if len(widget_value) > 0:
                value = type(widget_value)
            setattr(self, attribute_name, value)
            getattr(self.w, "label_" + attribute_name).set_text(str(value))
            return True

        return callback

    def build_gui(self, container):
        """
        This method is called when the plugin is invoked.  It builds the
        GUI used by the plugin into the widget layout passed as
        ``container``.
        This method may be called many times as the plugin is opened and
        closed for modal operations.  The method may be omitted if there
        is no GUI for the plugin.

        This specific example uses the GUI widget set agnostic wrappers
        to build the GUI, but you can also just as easily use explicit
        toolkit calls here if you only want to support one widget set.
        """
        top = Widgets.VBox()
        top.set_border_width(4)

        # this is a little trick for making plugins that work either in
        # a vertical or horizontal orientation.  It returns a box container,
        # a scroll widget and an orientation ('vertical', 'horizontal')
        vbox, scroll_widget, orientation = Widgets.get_oriented_box(container)
        vbox.set_border_width(4)
        vbox.set_spacing(2)

        # Take a text widget to show some instructions
        self.msg_font = self.fv.get_font("sansFont", 12)
        tw = Widgets.TextArea(wrap=True, editable=False)
        tw.set_font(self.msg_font)
        self.tw = tw

        # Frame for instructions and add the text widget with another
        # blank widget to stretch as needed to fill emp
        frame = Widgets.Expander("Instructions")
        frame.set_widget(tw)
        vbox.add_widget(frame, stretch=0)

        frame = Widgets.Frame("Region")
        hbox = Widgets.HBox()
        widget, bunch = Widgets.build_info(
            (
                ("X center:", "label", "label X center", "label", "X center", "entry"),
                ("Y center:", "label", "label Y center", "label", "Y center", "entry"),
                ("Center value:", "label", "label center value", "label"),
                ("RA center:", "label", "label RA center", "label"),
                ("Dec center:", "label", "label Dec center", "label"),
            )
        )
        self.w.update(bunch)
        self.w.label_x_center.set_text("")
        self.w.label_y_center.set_text("")
        self.w.label_center_value.set_text("")
        self.w.label_ra_center.set_text("")
        self.w.label_dec_center.set_text("")
        hbox.add_widget(widget)

        widget, bunch = Widgets.build_info(
            (
                ("Use view center", "button"),
                ("Centroid", "button"),
            )
        )

        def use_view_center_callback(widget) -> None:
            """Use the field of view center for the enhancement center."""

            if self.region is not None:
                self.region = None

            x, y = self.fv.get_viewer(self.chname).get_pan()

            DrawClass = self.canvas.get_draw_class(self.region_type)
            tag: str = self.canvas.add(
                DrawClass(
                    x,
                    y,
                    self.region_width // 2,
                    self.region_height // 2,
                    color=self.region_color,
                )
            )

            self.draw_callback(self.canvas, tag)

        bunch.use_view_center.set_tooltip("Center region in the view")
        bunch.use_view_center.add_callback("activated", use_view_center_callback)

        def centroid_callback(widget) -> None:
            """Centroid at the current region x, y position."""
            if self.region is None:
                return

            self.recenter_region_peak()

        bunch.centroid.add_callback("activated", centroid_callback)

        self.w.update(bunch)
        hbox.add_widget(widget)

        frame.set_widget(hbox)
        vbox.add_widget(frame)

        # # Astrometry tabs
        tab_widget = Widgets.TabWidget()
        self.w.tab_widget = tab_widget

        # Metadata tab
        metadata_vbox = Widgets.VBox()
        widget = (
            (
                "Target:",
                "label",
                "label_target",
                "label",
                "target",
                "entry",
            ),
            (
                "Observer location:",
                "label",
                "label_observer_location",
                "label",
                "observer location",
                "entry",
            ),
        )
        widget, bunch = Widgets.build_info(widget)
        self.w.update(bunch)

        # Metadata tab: object
        change_target = self.create_entry_callback("target")
        bunch.label_target.set_text(str(self.target))
        bunch.target.add_callback("activated", change_target)

        # Metadata tab: observer location
        change_observer_location = self.create_entry_callback("observer_location")
        bunch.label_observer_location.set_text(str(self.observer_location))
        bunch.observer_location.add_callback("activated", change_observer_location)

        metadata_vbox.add_widget(widget)
        tab_widget.add_widget(metadata_vbox, title="Metadata")

        # Settings tab
        settings_vbox = Widgets.VBox()
        widget = (
            (
                "Region type:",
                "label",
                "label_region_type",
                "label",
                "region type",
                "combobox",
            ),
            (
                "Max region size:",
                "label",
                "label_max_region_size",
                "label",
                "Max region size",
                "spinbutton",
            ),
            (
                "Centering method:",
                "label",
                "label_centering_method",
                "label",
                "Centering method",
                "combobox",
            ),
        )
        widget, bunch = Widgets.build_info(widget, orientation=orientation)
        self.w.update(bunch)

        # Settings tab: region type
        def change_region_type(w, index):
            self.region_type = self.region_types[index]
            self.w.label_region_type.set_text(self.region_type)
            self.canvas.set_drawtype(self.region_type, color="cyan", linestyle="dash")
            return True

        combobox = bunch.region_type
        for name in self.region_types:
            combobox.append_text(name)

        index = self.region_types.index(self.region_type)
        combobox.set_index(index)
        bunch.label_region_type.set_text(self.region_type)
        combobox.add_callback("activated", change_region_type)

        # Settings tab: max region size
        bunch.max_region_size.set_limits(5, 10000, incr_value=10)
        bunch.max_region_size.set_value(self.max_region_size)
        bunch.label_max_region_size.set_text(str(self.max_region_size))
        change_max_region_size = self.create_change_value_callback(
            "max_region_size", int
        )
        bunch.max_region_size.add_callback("value-changed", change_max_region_size)

        settings_vbox.add_widget(widget)
        tab_widget.add_widget(settings_vbox, title="Settings")
        vbox.add_widget(tab_widget)

        # Settings tab: centering method
        def change_centering_method(w, index):
            self.centering_method = self.centering_methods[index]
            self.w.label_centering_method.set_text(self.centering_method)
            return True

        combobox = bunch.centering_method
        for name in self.centering_methods:
            combobox.append_text(name)

        index = self.centering_methods.index(self.centering_method)
        combobox.set_index(index)
        bunch.label_centering_method.set_text(self.centering_method)
        combobox.add_callback("activated", change_centering_method)

        # scroll bars will allow lots of content to be accessed
        top.add_widget(scroll_widget, stretch=1)

        # A button box that is always visible at the bottom
        buttons = Widgets.HBox()
        buttons.set_spacing(3)

        # Add a close button for the convenience of the user
        button = Widgets.Button("Close")
        button.add_callback("activated", lambda w: self.close())
        buttons.add_widget(button, stretch=0)
        buttons.add_widget(Widgets.Label(""), stretch=1)
        top.add_widget(buttons, stretch=0)

        # Add our GUI to the container
        container.add_widget(top, stretch=1)

    def draw_callback(self, canvas, tag: str):
        """Create a new region based on the shape referenced by tag.

        If the shape is not an allowed region type, then the shape is deleted,
        and nothing is created.

        """

        shape: CanvasObjectBase = canvas.get_object_by_tag(tag)
        canvas.delete_object_by_tag(tag)

        if shape.kind not in self.region_types:
            return True

        if self.region is not None:
            self.region = None

        self.region = CenteringRegion(shape, self.fitsimage, self.canvas)
        self.recenter_region_peak()

    def move_region(self, x: float, y: float) -> None:
        """Move the region to x, y and update the image data."""
        if self.region is None:
            return

        self.region.set_center_point(x, y)
        self.region.update_image_data()

    def move_region_peak(self, x: float, y: float) -> None:
        """Move the region's center point and update the labels."""
        self.w.label_x_center.set_text(f"{x:.3f}")
        self.w.label_y_center.set_text(f"{y:.3f}")
        self.region.set_peak_point(x, y)
        self.w.label_center_value.set_text(f"{self.region.get_peak_value():.6g}")

        vip = self.fitsimage.get_vip()
        image, pt = vip.get_image_at_pt((x, y))
        try:
            ra_deg, dec_deg = image.pixtoradec(x, y, coords="data")
            ra_txt, dec_txt = wcs.deg2fmt(ra_deg, dec_deg, "str")
        except Exception as e:
            self.logger.warning("Couldn't calculate sky coordinates: %s" % (str(e)))
            ra_deg, dec_deg = 0.0, 0.0
            ra_txt = dec_txt = "BAD WCS"

        self.w.label_ra_center.set_text(ra_txt)
        self.w.label_dec_center.set_text(dec_txt)

    def recenter_region_peak(self) -> None:
        """Re-center the region peak."""
        x, y = self.region.centroid(self.w.centering_method.get_text())
        self.move_region_peak(x, y)

    def edit_callback(self, canvas, obj):
        if obj.kind not in self.region_types:
            return True

        if self.region is None:
            return True

        if self.region.canvas_object.has_object(obj):
            self.region.update_image_data()
            self.recenter_region_peak()

        return True

    def button_down(self, canvas, event, x, y, viewer) -> bool:
        """Move the region to the selected location, or create a new region."""

        if self.region is None:
            self.region = CenteringRegion.at_location(
                x,
                y,
                self.fitsimage,
                self.canvas,
                width=self.region_width,
                height=self.region_height,
                color=self.region_color,
            )
        else:
            self.region.set_center_point(x, y)

        return True

    def button_up(self, canvas, event, x, y, viewer) -> bool:
        """Move the region to the final location, update region data, and centroid."""

        if self.region is None:
            return False

        self.move_region(x, y)
        self.recenter_region_peak()

        return True

    def button_drag(self, canvas, event, x, y, viewer) -> bool:
        """Move the region with the button and re-center."""

        if self.region is None:
            return False

        self.region.set_center_point(x, y)

        return True

    def close(self) -> bool:
        """
        Example close method.  You can use this method and attach it as a
        callback to a button that you place in your GUI to close the plugin
        as a convenience to the user.
        """
        self.fv.stop_local_plugin(self.chname, str(self))
        return True

    def start(self):
        """
        This method is called just after ``build_gui()`` when the plugin
        is invoked.  This method may be called many times as the plugin is
        opened and closed for modal operations.  This method may be omitted
        in many cases.
        """

        self.tw.set_text(
            """Draw a region with the right mouse button or create a default """
            """region with the left mouse button.  Move an existing region """
            """with the left mouse button."""
        )

        # insert layer if it is not already
        p_canvas = self.fitsimage.get_canvas()
        try:
            p_canvas.get_object_by_tag(self.layer_tag)
        except KeyError:
            # Add canvas layer
            p_canvas.add(self.canvas, tag=self.layer_tag)

        self.resume()

    def pause(self):
        """
        This method is called when the plugin loses focus.
        It should take any actions necessary to stop handling user
        interaction events that were initiated in ``start()`` or
        ``resume()``.
        This method may be called many times as the plugin is focused
        or defocused.  It may be omitted if there is no user event handling
        to disable.
        """

        self.canvas.ui_set_active(False)

    def resume(self):
        """
        This method is called when the plugin gets focus.
        It should take any actions necessary to start handling user
        interaction events for the operations that it does.
        This method may be called many times as the plugin is focused or
        defocused.  The method may be omitted if there is no user event
        handling to enable.
        """

        self.canvas.ui_set_active(True, viewer=self.fitsimage)
        self.fv.show_status("Draw a region with the right mouse button")

    def stop(self):
        """
        This method is called when the plugin is stopped.
        It should perform any special clean up necessary to terminate
        the operation.  The GUI will be destroyed by the plugin manager
        so there is no need for the stop method to do that.
        This method may be called many times as the plugin is opened and
        closed for modal operations, and may be omitted if there is no
        special cleanup required when stopping.
        """

        # deactivate the canvas
        self.canvas.ui_set_active(False)
        p_canvas = self.fitsimage.get_canvas()
        try:
            p_canvas.delete_object_by_tag(self.layer_tag)
        except Exception:
            pass
        self.fv.show_status("")

    def redo(self):
        """
        This method is called when the plugin is active and a new
        image is loaded into the associated channel.  It can optionally
        redo the current operation on the new image.  This method may be
        called many times as new images are loaded while the plugin is
        active.  This method may be omitted.
        """

    def __str__(self):
        """
        This method should be provided and should return the lower case
        name of the plugin.
        """
        return "astrometry"
