"""Exercise the real waveform pointer handlers without a browser dependency."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_waveform_clicks_seek_and_only_drags_change_boundaries():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for waveform JavaScript regression checks")
    template = (Path(__file__).parents[1] / "templates" / "index.html").read_text()
    handlers = template.split("    const waveformTime =", 1)[1].split(
        "    window.addEventListener('resize'", 1
    )[0]
    harness = r"""
const assert = require('node:assert/strict');
const listeners = {};
const waveform = {
    getBoundingClientRect: () => ({left:100,width:1000}),
    addEventListener: (name, handler) => { listeners[name] = handler; },
    setPointerCapture: () => {},
};
const audio = {currentTime:0};
const $ = id => id === 'waveform' ? waveform : audio;
let updates=0;
const updateMarkers = () => { updates++; };
const drawWaveform = () => {};
let draggedMarker=null;
let viewStart=0;
let span=1000;
const viewSpan=()=>span;
const state={review:{audio_duration:1000,include_dynamic:true},manualTeaser:true,
    markers:{sermon_start:100,sermon_end:900,teaser_start:300,teaser_end:350},
    cut:{enabled:true,start:500,end:550},previewStop:999};
const fire = (type,x,extra={}) => listeners[type]({clientX:x,clientY:50,
    pointerId:1,button:0,isPrimary:true,currentTarget:waveform,preventDefault(){},...extra});
"""
    checks = r"""
const original = JSON.stringify({markers:state.markers,cut:state.cut});
for (const time of [100,900,300,350,500,550]) {
    // On a handle, near either side of it, and with normal hand jitter.
    for (const offset of [0,-10,10]) {
        const x=100+time+offset;
        fire('pointerdown',x);
        assert.equal(audio.currentTime,time+offset);
        fire('pointermove',x+2);
        fire('pointerup',x+2);
        assert.equal(audio.currentTime,time+offset+2);
        assert.equal(JSON.stringify({markers:state.markers,cut:state.cut}),original);
    }
}
assert.equal(updates,0);
assert.equal(state.previewStop,null);
fire('pointerdown',800); // Open waveform space also seeks.
fire('pointerup',800);
assert.equal(audio.currentTime,700);
for (const [collection,key] of [
    ['markers','sermon_start'],['markers','sermon_end'],
    ['markers','teaser_start'],['markers','teaser_end'],['cut','start'],['cut','end']
]) {
    const before=state[collection][key];
    const x=100+before+8;
    fire('pointerdown',x);
    fire('pointermove',x+20);
    fire('pointerup',x+25);
    assert.equal(state[collection][key],before+25); // No grab-offset snap.
    state[collection][key]=before;
}
// A different finger and cancelled gestures cannot change a pending marker.
fire('pointerdown',200);
fire('pointermove',250,{pointerId:2});
fire('pointercancel',200);
fire('pointermove',250);
fire('pointerup',250);
assert.equal(state.markers.sermon_start,100);
// Zoom and scroll use the same seek mapping and drag distance.
viewStart=80; span=100;
fire('pointerdown',310); // 101 seconds, near the 100-second boundary.
fire('pointerup',310);
assert.equal(audio.currentTime,101);
assert.equal(state.markers.sermon_start,100);
fire('pointerdown',310);
fire('pointermove',330);
fire('pointerup',330);
assert.equal(state.markers.sermon_start,102);
"""
    result = subprocess.run(
        [node, "-e", harness + "\nconst waveformTime =" + handlers + checks],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
