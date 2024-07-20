# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A Ginga local plugin for measuring small solar system object astrometry.


Based on Ginga's Pick tool.

"""

__all__ = ["Astrometry"]

from packaging.version import Version
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.time import Time

from ginga.GingaPlugin import LocalPlugin
from ginga.gw import Widgets
from ginga import ImageView
from ginga.rv.Control import GingaShell
from ginga.canvas.CanvasObject import CanvasObjectBase
from ginga.canvas.types.layer import DrawingCanvas
from ginga.canvas.types.basic import Point, Text
from ginga.misc import Bunch
from ginga.util import wcs
from ginga.util.vip import ViewerImageProxy
from ginga import __version__ as ginga_version
from sbpy.utils import optional_packages

try:
    from photutils.centroids import centroid_2dg
except ImportError:
    centroid_2dg = None


class RegionImageBoundsError(Exception):
    "A value is outside the region's image data."


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

    def get_center(self) -> Tuple[float]:
        """Get the region's center.


        Returns
        -------
        x, y : float

        """

        return self.shape.get_center_pt()

    def set_center(self, x: float, y: float) -> None:
        """Move the region to this center."""

        self.shape.move_to_pt((x, y))
        self.update_image_data()
        self.canvas.update_canvas()

    def get_center_point(self) -> Tuple[float, float]:
        """Get the center marker's coordinates.


        Returns
        -------
        x, y : float

        """

        return self.peak.get_center_pt()

    def set_center_point(self, x: float, y: float) -> None:
        """Set the center point's coordinates.


        Raises
        """

        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = self.get_llur()

        invalid_coordinates = any(
            [
                x < x1,
                x > x2,
                y < y1,
                y > y2,
                ~np.isfinite(x),
                ~np.isfinite(y),
            ]
        )

        if invalid_coordinates:
            self.peak.move_to_pt(self.get_center())
            self.peak.alpha = 0
            self.canvas.update_canvas()
            raise RegionImageBoundsError(
                f"Requested center ({x:.1f}, {y:.1f}) is outside of the image data."
            )

        self.peak.alpha = 1
        self.peak.move_to_pt((x, y))
        self.canvas.update_canvas()

    def get_center_point_value(self) -> float:
        """Image value at the center pixel."""

        if self.data.size == 0:
            return 0

        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = self.get_llur()

        x: float
        y: float
        x, y = self.get_center_point()

        invalid_coordinates = any(
            [
                x < x1,
                x > x2,
                y < y1,
                y > y2,
                ~np.isfinite(x),
                ~np.isfinite(y),
            ]
        )
        if invalid_coordinates:
            raise RegionImageBoundsError(
                f"Center point ({x}, {y}) is outside the region image."
            )

        x -= x1
        y -= y1

        x = int(round(x))
        y = int(round(y))

        return float(self.data[y, x])

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


class AstrometricReport:
    """Astrometry and associated metadata."""

    def __init__(self, tree_view):
        self.tree_view: Widgets.TreeView = tree_view
        self._report: Dict[str, Dict[str, Any]] = {}

        columns: List[Tuple[str, str]] = [
            ("Channel", "channel"),
            ("Name", "name"),
            ("Target", "target"),
            ("Date", "date"),
            ("Location", "location"),
            ("x", "x"),
            ("y", "y"),
            ("RA", "ra"),
            ("Dec", "dec"),
        ]
        self.tree_view.setup_table(columns, 1, "name")

    def clear(self) -> None:
        self.tree_view.clear()
        self._report = {}

    def save(self, filename: str) -> None:
        """Write ECSV formatted table to this file name."""
        tab = Table([row for row in self._report.values()])
        tab["ra"].unit = "deg"
        tab["dec"].unit = "deg"
        tab.meta["creator"] = "sbpy-ginga Astrometry tool"
        tab.meta["creation date"] = Time.now().iso
        tab.write(filename, format="ascii.ecsv", overwrite=True)

    def update(self, results: Dict[str, Any]) -> None:
        self._report.update(results)
        self.tree_view.set_tree(self._report)


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

        self.fv: GingaShell
        self.fitsimage: ImageView
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

        # header keywords for metadata, include an empty string in each list
        self.date_keywords: List[str] = sorted(
            [
                "",
                "OBSDATE",
                "DATEOBS",
                "OBS-DATE",
                "DATE-OBS",
            ]
        )

        self.target_keywords: List[str] = sorted(
            [
                "",
                "TARGET",
                "OBJECT",
            ]
        )

        # get preferences
        prefs = self.fv.get_preferences()
        self.settings = prefs.create_category("plugin_sbpy_Astrometry")
        self.settings.load(onError="silent")

        self.sync_preferences()

        self.region_width = CenteringRegion.region_default_width
        self.region_height = CenteringRegion.region_default_height

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

    @property
    def autofill_target(self) -> bool:
        return self.w.autofill_target.get_state()

    @autofill_target.setter
    def autofill_target(self, value: bool):
        self.w.autofill_target.set_state(value)

    @property
    def autofill_date(self) -> bool:
        return self.w.autofill_date.get_state()

    @autofill_date.setter
    def autofill_date(self, value: bool):
        self.w.autofill_date.set_state(value)

    @property
    def auto_levels(self) -> bool:
        return self.w.auto_levels.get_state()

    @auto_levels.setter
    def auto_levels(self, value: bool):
        self.w.auto_levels.set_state(value)

    @property
    def center_dec(self) -> Union[Angle, None]:
        return self._dec_center

    @center_dec.setter
    def center_dec(self, value: Union[Angle, None]):
        self._dec_center = value
        s = ""
        if value is not None:
            s = self._dec_center.to_string("deg", sep=":", precision=2)

        self.w.label_dec_center.set_text(s)

    @property
    def center_ra(self) -> Union[Angle, None]:
        return self._ra_center

    @center_ra.setter
    def center_ra(self, value: Union[Angle, None]):
        self._ra_center = value
        s = ""
        if value is not None:
            s = self._ra_center.to_string("hourangle", sep=":", precision=2)

        self.w.label_ra_center.set_text(s)

    @property
    def center_value(self) -> Union[float, None]:
        return self._center_value

    @center_value.setter
    def center_value(self, value: Union[float, None]):
        self._center_value = value
        s = ""
        if value is not None:
            s = f"{value:.6g}"

        self.w.label_center_value.set_text(s)

    @property
    def center_x(self) -> Union[float, None]:
        return self._x_center

    @center_x.setter
    def center_x(self, value: Union[float, None]):
        self._x_center = value
        s = ""
        if value is not None:
            s = f"{value:.3f}"

        self.w.label_x_center.set_text(s)

    @property
    def center_y(self) -> Union[float, None]:
        return self._y_center

    @center_y.setter
    def center_y(self, value: Union[float, None]):
        self._y_center = value
        s = ""
        if value is not None:
            s = f"{value:.3f}"

        self.w.label_y_center.set_text(s)

    @property
    def date(self) -> str:
        return self.w.label_date.get_text()

    @date.setter
    def date(self, value):
        self.w.label_date.set_text(str(value))

    @property
    def date_keyword(self) -> str:
        return self.w.label_date_keyword.get_text()

    @date_keyword.setter
    def date_keyword(self, keyword: str):
        keyword = keyword.upper()

        self.w.label_date_keyword.set_text(keyword)

    @property
    def observer_location(self) -> str:
        return self.w.label_observer_location.get_text()

    @observer_location.setter
    def observer_location(self, location: str):
        self.w.label_observer_location.set_text(location)

    @property
    def target(self) -> str:
        return self.w.label_target.get_text()

    @target.setter
    def target(self, value):
        self.w.label_target.set_text(str(value))

    @property
    def target_keyword(self) -> str:
        return self.w.label_target_keyword.get_text()

    @target_keyword.setter
    def target_keyword(self, keyword: str):
        keyword = keyword.upper()

        self.w.label_target_keyword.set_text(keyword)

        if keyword not in self.target_keywords:
            self.target_keywords.append(keyword)
            self.w.target_keyword_combobox.append_text(keyword)
        self.w.target_keyword_combobox.set_text(keyword)

    def _create_change_value_callback(self, attribute_name, type=str):
        """Create a basic callback that updates a value and associated label."""

        def callback(widget, value):
            self.w[attribute_name].set_text(str(type(value)))
            self.w["label_" + attribute_name].set_text(str(self.w[attribute_name]))
            return True

        return callback

    def _create_entry_callback(self, attribute_name, type=str):
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
            self.w[f"{attribute_name}_entry"].set_text(value)
            self.w[f"label_{attribute_name}"].set_text(str(value))
            return True

        return callback

    def _setup_header_autofill(self, bunch: Bunch, name: str) -> None:
        """Setup a checkbox, entry, and combobox to autofill a header value."""

        autofill: str = f"autofill_{name}"
        header_keyword: str = f"{name}_keyword"
        keywords: List[str] = getattr(self, f"{name}_keywords")
        combobox: Widgets.ComboBox = bunch[f"{name}_keyword_combobox"]

        bunch[f"label_{name}_keyword"].set_text("")
        bunch[autofill].set_state(False)
        for k in keywords:
            combobox.append_text(k)

        def activate_callback(widget, *args):
            keyword: str = widget.get_text()
            setattr(self, header_keyword, keyword)
            setattr(self, autofill, keyword != "")
            self.auto_update_metadata(autofill)
            return True

        if Version(ginga_version) < Version("5.2.0-dev6"):
            self.w[f"{name}_keyword_entry"].add_callback("activated", activate_callback)

        combobox.add_callback("activated", activate_callback)

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
                ("Add", "button"),
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

            self.recenter_region()

        bunch.centroid.add_callback("activated", centroid_callback)
        bunch.add.set_tooltip("Add this position to the report")

        self.w.update(bunch)
        hbox.add_widget(widget)

        frame.set_widget(hbox)
        vbox.add_widget(frame)

        # # Astrometry tabs
        tab_widget = Widgets.TabWidget()
        self.w.tab_widget = tab_widget

        # Metadata tab
        metadata_tab = Widgets.VBox()
        widget = (
            (
                "Target:",
                "label",
                "label_target",
                "label",
                "target_entry",
                "entry",
            ),
            (
                "Observer location:",
                "label",
                "label_observer_location",
                "label",
                "observer_location_entry",
                "entry",
            ),
            (
                "Date:",
                "label",
                "label_date",
                "label",
                "date_entry",
                "entry",
            ),
        )
        widget, bunch = Widgets.build_info(widget)
        self.w.update(bunch)

        # Metadata tab: target
        bunch.label_target.set_text("")
        change_target_callback = self._create_entry_callback("target")
        bunch.target_entry.add_callback("activated", change_target_callback)

        # Metadata tab: observer location
        bunch.label_observer_location.set_text("")
        change_observer_location_callback = self._create_entry_callback(
            "observer_location"
        )
        bunch.observer_location_entry.add_callback(
            "activated", change_observer_location_callback
        )

        # Metadata tab: date
        bunch.label_date.set_text("")
        change_date_callback = self._create_entry_callback("date")
        bunch.date_entry.add_callback("activated", change_date_callback)

        metadata_tab.add_widget(widget)
        tab_widget.add_widget(metadata_tab, title="Metadata")

        # Report tab
        report_tab: Widgets.VBox = Widgets.VBox()
        self.w.report_table = Widgets.TreeView(sortable=True)
        self.report = AstrometricReport(self.w.report_table)

        def add_to_report_callback(widget) -> None:
            """Add this position to the report."""

            if self.region is None:
                return

            image = self.fitsimage.get_image()
            if image is None:
                return
            image_name: str = image.get("name", "")

            data: Dict[str, Any] = {
                "channel": self.fv.get_channel_name(self.fitsimage),
                "name": image_name,
                "target": self.target,
                "date": self.date,
                "location": self.observer_location,
                "x": "" if self.center_x is None else round(self.center_x, 3),
                "y": "" if self.center_y is None else round(self.center_y, 3),
                "ra": (
                    ""
                    if self.center_ra is None
                    else self.center_ra.to_string("deg", decimal=True, precision=6)
                ),
                "dec": (
                    ""
                    if self.center_dec is None
                    else self.center_dec.to_string("deg", decimal=True, precision=6)
                ),
            }

            results: Dict[str, Any] = {}
            results[image_name] = data

            self.report.update(results)

        def save_report_callback(widget) -> None:
            """Open file dialog and save report."""
            dialog = Widgets.SaveDialog(title="Save as...")
            filename: Union[str, None] = dialog.get_path()
            if filename is None:
                return  # cancel
            self.report.save(filename)

        button_box: Widgets.HBox = Widgets.HBox()
        self.w.report_add_button = Widgets.Button("Add")
        self.w.report_clear_button = Widgets.Button("Clear")
        self.w.report_save_button = Widgets.Button("Save")

        self.w.report_add_button.add_callback("activated", add_to_report_callback)
        self.w.add.add_callback("activated", add_to_report_callback)
        self.w.report_clear_button.add_callback(
            "activated", lambda widget: self.report.clear()
        )
        self.w.report_save_button.add_callback("activated", save_report_callback)

        button_box.add_widget(self.w.report_add_button)
        button_box.add_widget(self.w.report_clear_button)
        button_box.add_widget(self.w.report_save_button)

        # End report tab
        report_tab.add_widget(self.w.report_table, stretch=1)
        report_tab.add_widget(button_box)
        tab_widget.add_widget(report_tab, title="Report")

        # Settings tab
        settings_tab = Widgets.VBox()
        design = (
            (
                "Region type:",
                "label",
                "label_region_type",
                "label",
                "region_type",
                "combobox",
            ),
            (
                "Max region size:",
                "label",
                "label_max_region_size",
                "label",
                "max_region_size",
                "spinbutton",
            ),
            (
                "Centering method:",
                "label",
                "label_centering_method",
                "label",
                "centering_method",
                "combobox",
            ),
            (
                "Auto levels:",
                "label",
                "",
                "label",
                "auto_levels",
                "checkbox",
            ),
        )

        if Version(ginga_version) < Version("5.2.0.dev6"):
            design += (
                (
                    "Target keyword:",
                    "label",
                    "label_target_keyword",
                    "label",
                    "target_keyword_entry",
                    "entry",
                ),
                (
                    "",
                    "label",
                    "",
                    "label",
                    "Autofill target",
                    "checkbox",
                    "target_keyword_combobox",
                    "combobox",
                ),
                (
                    "Date keyword:",
                    "label",
                    "label_date_keyword",
                    "label",
                    "date_keyword_entry",
                    "entry",
                ),
                (
                    "",
                    "label",
                    "",
                    "label",
                    "Autofill date",
                    "checkbox",
                    "date_keyword_combobox",
                    "combobox",
                ),
            )
        else:
            design += (
                (
                    "Target keyword:",
                    "label",
                    "label_target_keyword",
                    "label",
                    "target_keyword_combobox",
                    "comboboxedit",
                    "Autofill target",
                    "checkbox",
                ),
                (
                    "Date keyword:",
                    "label",
                    "label_date_keyword",
                    "label",
                    "date_keyword_combobox",
                    "comboboxedit",
                    "Autofill date",
                    "checkbox",
                ),
            )
        widget, bunch = Widgets.build_info(design, orientation=orientation)
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
        change_max_region_size_callback = self._create_change_value_callback(
            "max_region_size", int
        )
        bunch.max_region_size.add_callback(
            "value-changed", change_max_region_size_callback
        )

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

        # Settings tab: auto levels
        bunch.auto_levels.set_tooltip(
            "Automatically scale the image levels to the min/max value of the region"
        )
        bunch.auto_levels.set_state(True)

        def activate_auto_levels(widget: Widgets.CheckBox, value: Any):
            self.set_cut_levels()

        bunch.auto_levels.add_callback("activated", activate_auto_levels)

        # Settings tab: autofill target
        self._setup_header_autofill(bunch, "target")

        # Settings tab: autofill date
        self._setup_header_autofill(bunch, "date")

        # End settings tab
        settings_tab.add_widget(widget)
        tab_widget.add_widget(settings_tab, title="Settings")

        # End tab widget
        vbox.add_widget(tab_widget)

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

        self.auto_update_metadata()

    def set_cut_levels(self) -> None:
        """Set the image levels based on the region."""

        if not self.auto_levels or self.region is None:
            return

        im: np.ma.MaskedArray = self.region.get_image_data()
        self.fitsimage.cut_levels(np.nanmin(im), np.nanmax(im))

    def move_region(self, x: float, y: float) -> None:
        """Move the region to x, y and update the image data."""

        if self.region is None:
            return

        self.region.set_center(x, y)
        self.region.update_image_data()
        self.set_cut_levels()

    def move_region_peak(self, x: float, y: float) -> None:
        """Move the region's center point and update the labels."""

        try:
            self.region.set_center_point(x, y)
        except RegionImageBoundsError as exception:
            self.center_x = None
            self.center_y = None
            self.center_value = None
            self.center_ra = None
            self.center_dec = None
            self.fv.show_error(str(exception), raisetab=False)
            return

        self.center_x = x
        self.center_y = y
        self.center_value = self.region.get_center_point_value()

        vip = self.fitsimage.get_vip()
        image, pt = vip.get_image_at_pt((x, y))
        try:
            ra, dec = image.pixtoradec(x, y, coords="data")
            self.center_ra = Angle(ra, "deg")
            self.center_dec = Angle(dec, "deg")
        except Exception as e:
            self.logger.warning("Couldn't calculate sky coordinates: %s" % (str(e)))
            self.center_ra = None
            self.center_dec = None

    def recenter_region(self) -> None:
        """Re-center the region peak."""

        x, y = self.region.centroid(self.w.centering_method.get_text())
        self.move_region_peak(x, y)

    def auto_update_metadata(self, name: Optional[str] = None):
        """Auto-update astrometric metadata.


        Parameters
        ----------
        name : str, optional
            Only update this item.

        """

        vip: ViewerImageProxy = self.fitsimage.get_vip()
        point = self.fv.get_viewer(self.chname).get_pan()
        image, point2 = vip.get_image_at_pt(point)

        keyword: str
        names: List[str] = []
        if name is None:
            names.extend(["autofill_target", "autofill_date"])
        else:
            names.append(name)

        def search_for_keyword(keywords: List[str], image) -> str:
            """Raises KeyError if no keyword is found in the image metadata."""
            keyword: str
            for keyword in keywords:
                if image.get_keyword(keyword, None) is not None:
                    return keyword
            raise KeyError

        def autofill(name: str, image) -> None:
            """Checks metadata for current keyword and updates attribute `name`.

            If the current keyword is "", the header will be searched for a
            valid keyword.

            """

            keyword: str = getattr(self, f"{name}_keyword")
            keywords: List[str] = getattr(self, f"{name}_keywords")

            try:
                if keyword == "":
                    keyword = search_for_keyword(keywords, image)
                    setattr(self, f"{name}_keyword", keyword)
                setattr(self, name, image.get_keyword(keyword))
            except KeyError:
                pass

        if "autofill_target" in names and self.autofill_target:
            autofill("target", image)

        if "autofill_date" in names and self.autofill_date:
            autofill("date", image)

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
            del self.region
            self.region = None

        self.region = CenteringRegion(shape, self.fitsimage, self.canvas)
        self.set_cut_levels()
        self.recenter_region()

    def edit_callback(self, canvas, obj) -> bool:
        if obj.kind not in self.region_types:
            return True

        if self.region is None:
            return True

        if self.region.canvas_object.has_object(obj):
            self.region.update_image_data()
            self.set_cut_levels()
            self.recenter_region()

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
            self.region.set_center(x, y)

        return True

    def button_up(self, canvas, event, x, y, viewer) -> bool:
        """Move the region to the final location, update region data, and centroid."""

        if self.region is None:
            return False

        self.move_region(x, y)
        self.recenter_region()

        return True

    def button_drag(self, canvas, event, x, y, viewer) -> bool:
        """Move the region with the button and re-center."""

        if self.region is None:
            return False

        self.region.set_center(x, y)

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

        self.region.update_image_data()
        self.auto_update_metadata()
        self.recenter_region()
        self.set_cut_levels()

    def __str__(self):
        """
        This method should be provided and should return the lower case
        name of the plugin.
        """
        return "astrometry"
