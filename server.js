// Import necessary packages
const express = require('express');
const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');

// --- CONFIGURATION ---
const app = express();
const port = 3000;
// Your connection string with the new password
const mongoUri = 'mongodb+srv://greatstack:Pushkar@cluster0.qmjdgql.mongodb.net/';
const dbName = 'blogDB'; // You can name your database anything
const postsCollectionName = 'posts';

// --- MIDDLEWARE ---
app.use(cors()); // Allows requests from your front-end
app.use(express.json()); // Allows the server to understand JSON data

// --- DATABASE CONNECTION ---
let db;
async function connectToDb() {
    try {
        const client = new MongoClient(mongoUri);
        await client.connect();
        db = client.db(dbName);
        console.log(`Successfully connected to MongoDB Atlas database: ${dbName}`);
    } catch (err) {
        console.error('Failed to connect to MongoDB Atlas', err);
        process.exit(1); // Exit the process if DB connection fails
    }
}

// --- API ROUTES ---

// GET all posts
app.get('/api/posts', async (req, res) => {
    try {
        const posts = await db.collection(postsCollectionName).find({}).sort({ date: -1 }).toArray();
        res.json(posts);
    } catch (err) {
        res.status(500).json({ message: 'Failed to retrieve posts', error: err });
    }
});

// CREATE a new post
app.post('/api/posts', async (req, res) => {
    try {
        const newPost = {
            ...req.body,
            date: new Date(), // Set the current date
            comments: [] // Initialize with an empty comments array
        };
        const result = await db.collection(postsCollectionName).insertOne(newPost);
        res.status(201).json(result);
    } catch (err) {
        res.status(500).json({ message: 'Failed to create post', error: err });
    }
});

// UPDATE a post
app.put('/api/posts/:id', async (req, res) => {
    try {
        const { id } = req.params;
        // Make sure not to update the _id field
        const { _id, ...updateDataPayload } = req.body;
        const updateData = { $set: updateDataPayload };
        const result = await db.collection(postsCollectionName).updateOne({ _id: new ObjectId(id) }, updateData);
        if (result.matchedCount === 0) {
            return res.status(404).json({ message: 'Post not found' });
        }
        res.json(result);
    } catch (err) {
        res.status(500).json({ message: 'Failed to update post', error: err });
    }
});


// DELETE a post
app.delete('/api/posts/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const result = await db.collection(postsCollectionName).deleteOne({ _id: new ObjectId(id) });
        if (result.deletedCount === 0) {
            return res.status(404).json({ message: 'Post not found' });
        }
        res.status(204).send(); // 204 No Content
    } catch (err) {
        res.status(500).json({ message: 'Failed to delete post', error: err });
    }
});

// CREATE a new comment on a post
app.post('/api/posts/:id/comments', async (req, res) => {
    try {
        const { id } = req.params;
        const newComment = {
            id: new ObjectId(), // Generate a unique ID for the comment
            ...req.body,
            status: 'approved',
            replies: []
        };
        const result = await db.collection(postsCollectionName).updateOne(
            { _id: new ObjectId(id) },
            { $push: { comments: newComment } }
        );
        if (result.matchedCount === 0) {
            return res.status(404).json({ message: 'Post not found' });
        }
        // Fetch and return the updated post
        const updatedPost = await db.collection(postsCollectionName).findOne({ _id: new ObjectId(id) });
        res.status(201).json(updatedPost);

    } catch (err) {
        res.status(500).json({ message: 'Failed to add comment', error: err });
    }
});

// --- START SERVER ---
connectToDb().then(() => {
    app.listen(port, () => {
        console.log(`Server is running on http://localhost:${port}`);
    });
});

