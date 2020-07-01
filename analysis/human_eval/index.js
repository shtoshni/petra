/*jslint browser: true*/
/*global $, jQuery, alert*/
/*global d3*/
/*jslint node: true */
'use strict';

function transpose(matrix) {
  return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

function checkPrediction(g_coref, prediction){
    if (g_coref == prediction){
        return "#fff";
    }
    else {
        return "pink";
    }
}

function getMax(a){
  return Math.max(...a.map(e => Array.isArray(e) ? getMax(e) : e));
}

function getMin(a){
  return Math.min(...a.map(e => Array.isArray(e) ? getMin(e) : e));
}

function drawMemoryLog(dataset, model){
    // DRAW PLOTS
    let n = dataset["model_A_ent"].length;
    let x = [];
    let counter = 0;
    while (counter < n){
        x.push(counter);
        counter += 1;
    }

    var over_data = transpose(dataset["model_" + model + "_overwrite"]);
    var coref_data = transpose(dataset["model_" + model + "_coref"]);
    var comb_data = over_data.concat(coref_data);

    var max_val = getMax(comb_data);
    var min_val = getMin(comb_data);

    var num_cells = over_data.length;
    var range_arr = [];
    counter = 0;
    while (counter < num_cells){
        range_arr.push(counter);
        counter += 1;
    }

    var plot_data = [];

    for (let cell_idx in range_arr){
        cell_idx = parseInt(cell_idx);
        var start_idx = cell_idx;
        var end_idx = cell_idx + 1;
        let cur_cell_over = over_data.slice(start_idx, end_idx);
        let cur_cell_coref = coref_data.slice(start_idx, end_idx);
        let cur_cell_data = cur_cell_coref.concat(cur_cell_over);

        var colorscaleValue = [
            [0, '#0066CC'],
            [1, '#FFFF00']
            ];
        var cur_cell_plot_data = {
            z: cur_cell_data,
            x: x,
            type: 'heatmap',
            zmin: min_val,
            zmax: max_val,
            xaxis: 'x', // + (cell_idx + 1).toString(),
            yaxis: 'y' + (cell_idx + 1).toString(),
            name: "Cell " + (cell_idx + 1).toString(),
            hovertemplate:'<b>%{x}</b><br>%{y}<br>%{z:.2f}</b>',
            showscale: true,
            colorscale: 'Hot',
            colorbar: {
                showticklabels:  false,
                thickness: 15,
                xpad: 0,
            },
        };
        plot_data.push(cur_cell_plot_data);
    }

    var yaxis_labels = ['Coreference', 'New Person'];
    var yaxis_fmt = {
        dtick: 1,
        tickvals: [0, 1],
        ticktext: yaxis_labels,
    };
    var xaxis_fmt = {
        showticklabels: false,
        ticks: "",
    };
    var layout = {
        color: 'blue',
        showscale: false,
        showlegend: false,
        height: 700,
        dragmode: 'pan',
        grid: {
            rows: 8, columns: 1,
            // pattern: 'independent',
            subplots: ['xy1', 'xy2', 'xy3', 'xy4', 'xy5', 'xy6', 'xy7', 'xy8', ],
            roworder:'bottom to top',
        },

        title: {
            text: '<b>Model ' + model + ' - Memory Log</b>',
            color: 'lightgrey',
            fontsize: 18,
        },

        xaxis:{
            showticklabels: true,
            tickangle: 60,
            tickvals: x,
            ticktext: dataset["text"],
            tickfont: {
                size: 'auto',
            },
            rangeslider: {thickness: 0.05}
        },
        yaxis1: yaxis_fmt,
        yaxis2: yaxis_fmt,
        yaxis3: yaxis_fmt,
        yaxis4: yaxis_fmt,
        yaxis5: yaxis_fmt,
        yaxis6: yaxis_fmt,
        yaxis7: yaxis_fmt,
        yaxis8: yaxis_fmt,
    };

    let plot_id = "over_coref_" + model;
    d3.select("body").append("div").attr("id", plot_id);
    Plotly.newPlot(plot_id, plot_data, layout);

}

function drawPlots(initial_data, dataset){
    var p_elem = d3.select("body")
        .select("div")
        .append("p");
    p_elem.append("a")
        .html(initial_data.text);
    p_elem.style("font-size", "16px");
    var background_color = undefined;
    background_color = "white";

    drawMemoryLog(dataset, 'A');
    drawMemoryLog(dataset, 'B');
}

function showInitialData(initial_data, data){
    let id_list = [];
    for (var key in initial_data){
        id_list.push(parseInt(key, 10));
    }

    id_list.sort(function(a, b){return a-b});
    for (var i in id_list){
        let id = id_list[i].toString();
        var p_elem = d3.select("body")
            .select("div")
            .append("p");
        p_elem.append("a")
            .html(initial_data[id].text)
            .attr("target", "_blank")
            .attr("href", window.location.href + "?instance=" + id);
        var background_color = undefined;
        let errors =  data[id].errors;
        if (errors == 2){
            background_color = "#E9967A";
        } else if (errors == 1){
            background_color = "pink";
        } else {
            background_color = "white";
        }
        p_elem.style("background-color", background_color);
    }
}


$(document).ready(function () {
    if (window.location.href.indexOf('instance') > 1) {
        var url = window.location.href;
        var id = url.split('instance=')[1];
        drawPlots(initial_data[id], data[id]);
        // drawPlot(initial_data[id], data[id], 'B');
    }
    else {
        showInitialData(initial_data, data);
    }
}());
