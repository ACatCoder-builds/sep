//! Program to find queries in a text by 
//! embedding the query.
//!
//! Note that Args describes the program and not
//! the struct as it is used for parsing CLI args.
use anyhow::Result;

use regex::regex;

use std::time::Instant;
use std::io::Read;
use std::io::stdin;
use std::process::exit;

use fastembed::TextEmbedding;
use fastembed::TextInitOptions;
use fastembed::EmbeddingModel;
use fastembed::similarity::top_k;

use crossterm::style::Stylize;
use crossterm::style::StyledContent;

use clap::Parser;

/// This runs `run` and also prints any
/// errors that may occur.
fn main() {    
    if let Err(e) = run() {
	println!("sep: {e}");
	exit(1);
    }
}

/// A program that takes a query, embeds
/// them and then finds it in a text.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The query to embed and look for.
    query: String,
    /// The model that is used.
    /// Available models: <https://docs.rs/fastembed/latest/fastembed/enum.EmbeddingModel.html#variants>
    #[arg(short, long, default_value_t = EmbeddingModel::EmbeddingGemma300M)]
    model: EmbeddingModel,
    #[arg(short, long, default_value_t = 0.08)]
    /// The program checks if 1 or 2 spaces follow each period
    /// to attempt to determine if the text uses 2 spaces or 1 spaces
    /// after each point to end a sentence. The threshold using it
    /// can be configured, so if you know that each period at the
    /// end of a sentence is followed by 2 spaces, you can set this at
    /// 1, so it is guaranteed to split with 2 spaces instead of 1 space.
    /// Do not that the system may not catch periods that are not followed
    /// by some whitespace, e.g. citations. Please provide a value
    /// between 0 and 1
    punct_threshold: f64,
    #[arg(short, long)]
    /// If the program should output, useful when writing
    /// to a file.
    raw: bool,
}

/// Embeds and prints the result.
fn run() -> Result<()> {
    let args = Args::parse();
    
    let query = args.query;
    let mut text = Vec::new();
    stdin().read_to_end(&mut text)?;
    let text = chunk(
	&String::from_utf8(text)?,
	args.punct_threshold
    );
    
    let time = Instant::now();
    
    let mut model = TextEmbedding::try_new(
	TextInitOptions::new(args.model)
    )?;

    if !args.raw {
	println!("Currently embedding, this may take a while. :)");
    }
    
    let text_embedding = model.embed(&text, None)?;

    let query_embedding = model.embed([query], None)?;
    
    let query_embedding = &query_embedding[0];
    
    let results = top_k(query_embedding, &text_embedding, text_embedding.len());

    println!("Embedded succesfully in {:?}", time.elapsed());
    
    for (i, result) in results {
	if args.raw {
	    println!("{}: {}%\n", text[i], result * 100.);   
	} else {
	    println!("{}: {}%\n", text[i], color_percentages(result * 100.));   
	}
    }
    
    Ok(())
}

/// Takes an f32 and pretty prints it by giving it color.
///
/// These occur
/// as follows:
/// ```text
/// < 20% -> dark red
/// 20% to ~40% -> red
/// 40% to ~50% -> orange (dark yellow)
/// 50% to ~60% -> yellow
/// 60% to ~70% -> dark green
/// >= 70% -> green
/// ```
fn color_percentages(percentage: f32) -> StyledContent<String> {
    if percentage < 20. {
	return percentage
	    .to_string()
	    .dark_red()
    }
    if percentage < 40. {
	return percentage
	    .to_string()
	    .red();
    }
    if percentage < 50. {
	return percentage
	    .to_string()
	    .dark_yellow();
    }
    if percentage < 60. {
	return percentage
	    .to_string()
	    .yellow();
    }
    if percentage < 70. {
	return percentage
	    .to_string()
	    .dark_green();
    }
    
    percentage
        .to_string()
        .green()
}

/// Takes an input and a punctuation threshold
/// and creates a list of chunks to be fed into
/// embedding.
///
/// **NOTE**: When `punctuation` is written, I mean
/// periods, exclamation and question marks, not commas,
/// colons, semicolons, parenthesees, etc.
/// The algorithm goes as follows:
/// ```text
/// 1. get the total number of .?!
/// 2. get the total number of punctuation followed by 2 spaces or a newline.
/// 3. Check if the ratio of punctuation followed by 2 spaces / total punctuation
/// 4. If it is use that, else use just 1 space.
/// --
/// 1. Iterate through the sentences and note the last one.
/// 2. If there is only 1 or less sentence that were, previously logged,
/// log this one and continue
/// 3. If there are atleast 2 that were previously logged, merge those into a
/// chunk and clear the first instance.
/// ```
///
/// A demonstration where each letter is a sentence:
/// ```text
/// A B C D E F
/// ```
///
/// These would produce following chunks:
/// ```text
/// A B C
/// B C D
/// C D E
/// D E F
/// ```
fn chunk(input: &str, punct_threshold: f64) -> Vec<String> {
    
    let mut out = vec![];

    let dots = input
        .bytes()
        .filter(|c| *c == b'.' || *c == b'?' || *c == b'!')
        .count();

    let pattern = regex!(r"[.?!]  |[.?!]\n|[.?!]\r\n|[.?!]\r");

    let sentences = pattern.split(input);
    
    // Check for periods followed by 2 spaces.
    if (pattern.split(input).count() as f64 / dots as f64) < punct_threshold {
	
	let mut prev = vec![];

	
	for i in sentences {
	    
	    if prev.len() <= 1 {
		prev.push(i);
		continue;
	    }

	    
	    if prev.len() > 1 {
		out.push(format!("{}.  {}.  {}", prev[prev.len() - 2], prev[prev.len() -1], i));
		prev.remove(0);
	    }

	    prev.push(i);

	}
	
	return out;
    }

    // Check for dot followed by 1 space as fallback.
    
    let pattern = regex!(r"[.?!]\s");

    let sentences = pattern.split(input);


    
    let mut prev = vec![];
    
    for i in sentences {
	if prev.len() <= 1 {
	    prev.push(i);
	    continue;
	}
	
	if prev.len() > 1 {
	    out.push(format!("{}.  {}.  {}", prev[prev.len() - 2], prev[prev.len() -1], i));
	    prev.remove(0);
	}

	prev.push(i);

    }
    
    out
}
