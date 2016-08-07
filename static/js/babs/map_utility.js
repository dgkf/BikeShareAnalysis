$(document).ready(new function() {
  var INIT_ZOOM = 14

  /*
   *   Section 1
   *   Initialize leaflet maps
   */

  // Map #1: Usage (Plotting daily traffic)
  var usage_map = L.map('usage_map').setView([37.787, -122.405], INIT_ZOOM);
  L.tileLayer('https://api.mapbox.com/v4/mapbox.light/{z}/{x}/{y}.png?access_token={accessToken}', {
      attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
      maxZoom: 18,
      accessToken: 'pk.eyJ1IjoiZGdrZiIsImEiOiJjaXIzenM2dnIwMWtvZnBubW54MXF4Y2trIn0.GAEbsurvWsdMHL_RnPrO7w'
  }).addTo(usage_map);

  // Map #2: Predictions heatmap
  var prediction_map = L.map('prediction_map').setView([37.787, -122.405], INIT_ZOOM);
  L.tileLayer('https://api.mapbox.com/v4/mapbox.light/{z}/{x}/{y}.png?access_token={accessToken}', {
      attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
      maxZoom: 18,
      accessToken: 'pk.eyJ1IjoiZGdrZiIsImEiOiJjaXIzenM2dnIwMWtvZnBubW54MXF4Y2trIn0.GAEbsurvWsdMHL_RnPrO7w'
  }).addTo(prediction_map);



  /*
   *   Section 2
   *   Data loading and overlay plotting
   */

  // Nested data loading to ensure everything is loaded before drawn
  d3.json("static/json/station_locations.json", function(locations) {
    // Initialize SVG layer for usage map
    L.svg().addTo(usage_map);
    var usage_svg = d3.select("#usage_map").select("svg");
    var usage_g   = usage_svg.select("g");

    // Initialize SVG layer for prediction map
    L.svg().addTo(prediction_map);
    var prediction_svg = d3.select("#prediction_map").select("svg");
    var prediction_g   = prediction_svg.select("g");

    // Data Prep
    locations.forEach(function(d) {
      d.LatLng = new L.LatLng(d.Latitude, d.Longitude);
    })

    // function to set point size
    function getRadiusPx(map) {
      return Math.pow(2, (map.getZoom()-INIT_ZOOM+2))
    }

    // Aggregate Trip Data for usage map
    d3.json("static/json/trip_agg.json", function(trips) {
      var trip_lines = [];
      trips.forEach(function(d) {
        var slat = d.StartLatLng.Lat, slng = d.StartLatLng.Lng, elat = d.EndLatLng.Lat, elng = d.EndLatLng.Lng;
        var dist = Math.sqrt(Math.pow(elat-slat,2)+Math.pow(elng-slng,2));
        for (i=0; i < d.AvgDailyTrips; i++) {
          trip_lines.push({i: Math.floor(trip_lines.length / 4),
               latlng: new L.LatLng(slat, slng)});
          trip_lines.push({i: Math.floor(trip_lines.length / 4),
               latlng: new L.LatLng(
                 slat + (elat-slat)*0.33 + (elng-slng)*(i/d.AvgDailyTrips*0.04+0.005),
                 slng + (elng-slng)*0.33 - (elat-slat)*(i/d.AvgDailyTrips*0.04+0.005) )});
          trip_lines.push({i: Math.floor(trip_lines.length / 4),
               latlng: new L.LatLng(
                  slat + (elat-slat)*0.67 + (elng-slng)*(i/d.AvgDailyTrips*0.04+0.005),
                  slng + (elng-slng)*0.67 - (elat-slat)*(i/d.AvgDailyTrips*0.04+0.005) )});
          trip_lines.push({i: Math.floor(trip_lines.length / 4),
               latlng: new L.LatLng(elat, elng)});
          break;
        }
      })

      var trip_data = d3.nest()
        .key(function(d) { return d.i })
        .entries(trip_lines);

      // nodes
      var nodes = usage_g.selectAll("circle")
        .data(locations)
        .enter()
        .append("circle")
        .style("stroke", "black")
        .style("opacity", 0.8)
        .style("fill", "navy")
        .attr("r", getRadiusPx(usage_map))

      // lines
      // first some helper functions
      var line_from_latlng_array = d3.line()
        .x(function(d) { return usage_map.latLngToLayerPoint(d.latlng).x })
        .y(function(d) { return usage_map.latLngToLayerPoint(d.latlng).y })
        .curve(d3.curveCatmullRom.alpha(0.5))

      // Function for adding d3 transition - ultimately replaced by css animation
      /*
      function transition_wrapper() {
        transition_repeat(d3.select(this))

        function transition_repeat(path) {
          path.transition()
              .duration(15000)
              .ease(d3.easeLinear)
              .styleTween("stroke-dashoffset", function() { return d3.interpolateNumber(0, 499) })
              .on("end", function() { transition_repeat(path) } )
              .on("interrupt", function() { transition_repeat(path) } )
        }
      }
      */

      // create lines
      var lines = usage_g.selectAll(".series")
        .data(trip_data)
        .enter()
        .append("path")
        .attr("class", "tripline")
        .attr("d", function(d) { return line_from_latlng_array(d.values) })
        .style("fill", "none")
        .style("stroke", "#4444AA")
        .style("stroke-width", "3px")
        .style("stroke-opacity", 0.5)
        .style("stroke-dasharray", function(d) { i = Math.floor(Math.random() * 497);
                                                 return "0,"+i+",3,"+(500-i) })
        .style("stroke-dashoffset", 0)
        //.each(transition_wrapper) // Adds repetition
        .style("animation", function(d) {
                              return "dashoffset_animation " +
                                     (10.0 * 500 / this.getTotalLength()) +
                                     "s linear 0s infinite" });

      // set up callback for moving map
      usage_map.on("zoom", update);
      usage_map.on("viewreset", update);
      update();

      // update function for datapoints on map move
      function update() {
        // reposition nodes based on latlong offset
        nodes.attr("transform", function(d) {
          return "translate("+
            usage_map.latLngToLayerPoint(d.LatLng).x+","+
            usage_map.latLngToLayerPoint(d.LatLng).y+")";
          })
          .attr("r", getRadiusPx(usage_map));

        // reposition lines based on latlong offset
        lines.attr("d", function(d) { return line_from_latlng_array(d.values)})
      }
    });

    // Heatmap data
    d3.json("static/json/predictions.json", function(predictions) {
      // nodes
      var nodes = prediction_g.selectAll("circle")
        .data(locations)
        .enter()
        .append("circle")
        .style("stroke", "black")
        .style("opacity", 0.8)
        .style("fill", "navy")
        .attr("r", getRadiusPx(prediction_map))

      // set up callback for moving map
      prediction_map.on("zoom", update);
      prediction_map.on("viewreset", update);
      update();

      // update function for datapoints on map move
      function update() {
        // reposition nodes based on latlong offset
        nodes.attr("transform", function(d) {
          return "translate("+
            prediction_map.latLngToLayerPoint(d.LatLng).x+","+
            prediction_map.latLngToLayerPoint(d.LatLng).y+")";
          })
          .attr("r", getRadiusPx(prediction_map));
      }


      predictions = {data: predictions};
      var prediction_heatmap_layer = new HeatmapOverlay({
        "radius": 0.005,
        "maxOpacity": .4,
        "scaleRadius": true,
        "useLocalExtrema": true,
        latField: "Lat",
        lngField: "Lng",
        valueField: "Prediction"});

      prediction_heatmap_layer.addTo(prediction_map);
      prediction_heatmap_layer.setData(predictions);
    });
  });
});
