$(document).ready(new function() {
  var INIT_ZOOM = 14

  // Initialize our leaflet map
  var usage_map = L.map('usage_map').setView([37.787, -122.405], INIT_ZOOM);
  L.tileLayer('https://api.mapbox.com/v4/mapbox.light/{z}/{x}/{y}.png?access_token={accessToken}', {
      attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
      maxZoom: 18,
      accessToken: 'pk.eyJ1IjoiZGdrZiIsImEiOiJjaXIzenM2dnIwMWtvZnBubW54MXF4Y2trIn0.GAEbsurvWsdMHL_RnPrO7w'
  }).addTo(usage_map);

  // Initialize SVG layer
  L.svg().addTo(usage_map);

  // Pick SVG from map object
  var svg = d3.select("#usage_map").select("svg");
  var g   = svg.select("g");

  // Nested data loading to ensure everything is loaded before drawn
  d3.json("static/json/trip_agg.json", function(trips) {
    d3.json("static/json/station_locations.json", function(locations) {
      // Data Prep
      locations.forEach(function(d) {
        d.LatLng = new L.LatLng(d.Latitude, d.Longitude);
      })

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
      var nodes = g.selectAll("circle")
        .data(locations)
        .enter()
        .append("circle")
        .style("stroke", "black")
        .style("opacity", 0.8)
        .style("fill", "navy")
        .attr("r", getRadiusPx())

      // lines
      // first some helper functions
      var line_from_latlng_array = d3.line()
        .x(function(d) { return usage_map.latLngToLayerPoint(d.latlng).x })
        .y(function(d) { return usage_map.latLngToLayerPoint(d.latlng).y })
        .curve(d3.curveCatmullRom.alpha(0.5))
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
      // create lines
      var lines = g.selectAll(".series")
        .data(trip_data)
        .enter()
        .append("path")
        .attr("class", "tripline")
        .attr("d", function(d) { return line_from_latlng_array(d.values) })
        .style("fill", "none")
        .style("stroke", "#4444AA")
        .style("stroke-width", "3px")
        .style("stroke-opacity", 0.3)
        .style("stroke-dasharray", function(d) { i = Math.floor(Math.random() * 497);
                                                 return "0,"+i+",3,"+(500-i) })
        .style("stroke-dashoffset", 0)
        .each(transition_wrapper) // Adds repetition

      // set up callback for moving map
      usage_map.on("zoom", update);
      usage_map.on("viewreset", update);
      update();

      // function to set point size
      function getRadiusPx() {
        return Math.pow(2, (usage_map.getZoom()-INIT_ZOOM+2))
      }

      // update function for datapoints on map move
      function update() {
        // reposition nodes based on latlong offset
        nodes.attr("transform", function(d) {
          return "translate("+
            usage_map.latLngToLayerPoint(d.LatLng).x+","+
            usage_map.latLngToLayerPoint(d.LatLng).y+")";
          })
          .attr("r", getRadiusPx());

        // reposition lines based on latlong offset
        lines.attr("d", function(d) { return line_from_latlng_array(d.values)})
          .style("stroke-opacity", 0.3 * Math.pow(2, (usage_map.getZoom()-INIT_ZOOM)));
      }
    });
  });
});
